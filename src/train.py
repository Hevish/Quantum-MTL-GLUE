# train.py
import torch
import torch.nn as nn
import random
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from utils import compute_metrics
from tasks import GLUE_TASKS

def convert_batch_format(batch):
    """Convert single-task batch format to multi-task format expected by model"""
    return {
        "input1": batch["input_ids"],
        "input2": None,  # Single sentence tasks don't have input2
        "attention_mask1": batch["attention_mask"],
        "attention_mask2": None,  # Single sentence tasks don't have attention_mask2
        "labels": batch["labels"]
    }


def compute_task_loss(outputs, labels, task_name):
    """Compute appropriate loss based on task type"""
    # Handle dictionary output from model
    if isinstance(outputs, dict):
        logits = outputs['logits']
        # If model already computed loss, use it
        if 'loss' in outputs:
            return outputs['loss']
    else:
        logits = outputs

    # Compute loss based on task type
    if task_name == "stsb":
        return torch.nn.functional.mse_loss(logits.squeeze(), labels.float())
    elif task_name in ["cola", "sst2", "mrpc", "qnli", "qqp", "rte", "wnli"]:
        return torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(-1), labels.float())
    elif task_name == "mnli":
        return torch.nn.functional.cross_entropy(logits, labels)


def train_glue_baselines_style(model, dataloaders, task_names, device=None, num_epochs=3):
    """Train model on multiple GLUE tasks using task sampling"""

    # Ensure model is in training mode
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    # Create combined dataset from all tasks
    combined_batches = []
    for task_name in task_names:
        train_loader = dataloaders[task_name]["train"]
        for batch in train_loader:
            batch["task"] = task_name
            combined_batches.append(batch)

    total_steps = len(combined_batches) * num_epochs
    print(f"Total training steps: {total_steps}")

    model.train()  # Ensure training mode

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Shuffle combined batches each epoch
        import random
        random.shuffle(combined_batches)

        epoch_loss = 0
        num_batches = 0

        for step, batch in enumerate(combined_batches):
            # Ensure model is in training mode for each batch
            model.train()

            sampled_task = batch["task"]
            batch = convert_batch_format(batch)

            if device:
                batch_device = {}
                for k, v in batch.items():
                    if v is not None and hasattr(v, 'to'):
                        batch_device[k] = v.to(device)
                    else:
                        batch_device[k] = v
                batch = batch_device

            optimizer.zero_grad()

            # Forward pass with labels to compute loss internally
            outputs = model(
                task=sampled_task,
                input1=batch["input1"],
                input2=batch["input2"],
                attention_mask1=batch["attention_mask1"],
                attention_mask2=batch["attention_mask2"],
                label=batch["labels"]  # This triggers loss computation in model
            )
            # print("True train labels:", batch["labels"])
            loss = compute_task_loss(outputs, batch["labels"], sampled_task)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if step % 100 == 0:
                print(f"Step {step}, Task: {sampled_task}, Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        # Evaluate on validation sets
        print("Evaluating on validation sets...")
        for task_name in task_names:
            val_metrics = evaluate(model, dataloaders[task_name]["validation"], task_name, device=device)
            primary_metric = GLUE_TASKS[task_name]["metric"]
            score = val_metrics.get(primary_metric, val_metrics.get('accuracy', 0.0))
            print(f"{task_name}: {primary_metric} = {score:.4f}")

    return model

def evaluate(model, dataloader, task_name, device=None):
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            # Convert batch format first
            batch = convert_batch_format(batch)

            if device:
                # Handle None values properly when moving to device
                batch_device = {}
                for k, v in batch.items():
                    if v is not None and hasattr(v, 'to'):
                        batch_device[k] = v.to(device)
                    else:
                        batch_device[k] = v
                batch = batch_device

            outputs = model(
                task=task_name,
                input1=batch["input1"],
                input2=batch["input2"],
                attention_mask1=batch["attention_mask1"],
                attention_mask2=batch["attention_mask2"]
            )

            # Extract logits from dictionary output
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            # print("True labels:", batch["labels"])
            loss = compute_task_loss(outputs, batch["labels"], task_name)
            total_loss += loss.item()

            predictions.extend(logits.cpu().numpy())
            true_labels.extend(batch["labels"].cpu().numpy())

    # Calculate metrics based on task type
    task_config = GLUE_TASKS[task_name]

    if task_config["task_type"] == "regression":
        import numpy as np
        from scipy.stats import spearmanr
        predictions = np.array(predictions).squeeze()
        correlation, _ = spearmanr(predictions, true_labels)
        return {"spearmanr": correlation, "loss": total_loss / len(dataloader)}
    else:
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

        predictions = np.array(predictions)

        # Handle different task types
        if task_name in ["cola", "sst2", "mrpc", "qnli", "qqp", "rte", "wnli"]:
            # Binary classification with BCE - apply sigmoid and threshold
            from scipy.special import expit  # sigmoid function
            predictions = (expit(predictions.squeeze()) > 0.5).astype(int)
        else:
            # Multi-class classification (like MNLI) - use argmax
            predictions = np.argmax(predictions, axis=1)

        if task_config["metric"] == "accuracy":
            score = accuracy_score(true_labels, predictions)
            return {"accuracy": score, "loss": total_loss / len(dataloader)}
        elif task_config["metric"] == "f1":
            score = f1_score(true_labels, predictions, average="binary")
            return {"f1": score, "loss": total_loss / len(dataloader)}
        elif task_config["metric"] == "matthews_correlation":
            score = matthews_corrcoef(true_labels, predictions)
            return {"matthews_correlation": score, "loss": total_loss / len(dataloader)}