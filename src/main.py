# main.py
import torch
from model import MTLModel
from dataset import load_glue_task
from train import train_glue_baselines_style, evaluate
from tasks import GLUE_TASKS

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    task_names = ["cola", "sst2", "mrpc", "stsb", "mnli", "qnli", "qqp", "rte", "wnli"]

    # Load individual task dataloaders
    print("Loading individual task dataloaders...")
    dataloaders = {}
    for task in task_names:
        print(f"Loading {task}...")
        dataloaders[task] = load_glue_task(task, batch_size=128)

    # Initialize model
    selected_tasks = {k: v for k, v in GLUE_TASKS.items() if k in task_names}
    model = MTLModel(vocab_size=30522, tasks=selected_tasks).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train using GLUE-baselines approach
    print("\n=== Training (GLUE-baselines style) ===")
    model = train_glue_baselines_style(model, dataloaders, task_names, device=device)

    # Final evaluation
    print("\n=== FINAL RESULTS ===")
    final_results = {}
    task_scores = []

    for task in task_names:
        metrics = evaluate(model, dataloaders[task]["validation"], task, device=device)
        final_results[task] = metrics
        primary_metric = GLUE_TASKS[task]["metric"]
        score = metrics.get(primary_metric, metrics.get('accuracy', 0.0))
        task_scores.append(score)
        print(f"{task:>6}: {primary_metric} = {score:.4f}")

    macro_avg = sum(task_scores) / len(task_scores)
    print(f"\nMacro-average: {macro_avg:.4f}")

if __name__ == "__main__":
    main()