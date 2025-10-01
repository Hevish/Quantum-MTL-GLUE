# dataset.py
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional
from tasks import GLUE_TASKS


class GLUESingleTaskDataset(Dataset):
    """Single task dataset for GLUE-baselines approach"""

    def __init__(self, task_name: str, split="train", max_len=128, tokenizer_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.task_name = task_name
        self.task_config = GLUE_TASKS[task_name]

        self.data = self._load_task_data(split)
        print(f"Loaded {task_name} {split}: {len(self.data)} samples")

    def _load_task_data(self, split: str) -> List[Dict]:
        # Handle special cases for dataset loading
        if self.task_name == "mnli" and split == "validation":
            # For MNLI validation, we have both matched and mismatched
            dataset_matched = load_dataset("glue", self.task_name)["validation_matched"]
            dataset_mismatched = load_dataset("glue", self.task_name)["validation_mismatched"]
            dataset = list(dataset_matched) + list(dataset_mismatched)
            print(
                f"MNLI validation: {len(dataset_matched)} matched + {len(dataset_mismatched)} mismatched = {len(dataset)} total")
        else:
            dataset = load_dataset("glue", self.task_name)[split]

        task_data = []

        for sample in dataset:
            # Extract text based on task format
            text1, text2 = self._extract_texts(sample)

            # Skip invalid samples (only for test sets that have no labels)
            if sample.get("label", -1) == -1:
                continue

            # Tokenize
            encoding = self._tokenize_texts(text1, text2)

            # Handle labels
            label = self._process_label(sample["label"])

            task_data.append({
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": label
            })

        return task_data

    def _extract_texts(self, sample):
        """Extract text1 and text2 based on task type"""
        config = self.task_config

        if config["input_format"] == "single":
            if self.task_name == "cola":
                return sample["sentence"], None
            elif self.task_name == "sst2":
                return sample["sentence"], None
        else:  # pair format
            if self.task_name in ["mrpc", "stsb"]:
                return sample["sentence1"], sample["sentence2"]
            elif self.task_name == "mnli":
                return sample["premise"], sample["hypothesis"]
            elif self.task_name == "rte":
                return sample["sentence1"], sample["sentence2"]
            elif self.task_name == "qnli":
                return sample["question"], sample["sentence"]
            elif self.task_name == "qqp":
                return sample["question1"], sample["question2"]
            elif self.task_name == "wnli":
                return sample["sentence1"], sample["sentence2"]

        return sample.get("sentence", ""), None

    def _tokenize_texts(self, text1, text2):
        if text2 is None:
            return self.tokenizer(
                text1,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )
        else:
            return self.tokenizer(
                text1, text2,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )

    def _process_label(self, label):
        if self.task_name == "stsb":
            return float(label) / 5.0  # Normalize to [0,1]
        return int(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_glue_task(task_name: str, batch_size: int = 128, max_len: int = 128):
    """Load individual GLUE task dataloaders (GLUE-baselines approach)"""
    train_dataset = GLUESingleTaskDataset(task_name, split="train", max_len=max_len)
    # Use validation split instead of test for local evaluation
    val_dataset = GLUESingleTaskDataset(task_name, split="validation", max_len=max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return {"train": train_loader, "validation": val_loader}


# Keep multi-task classes for backward compatibility
class GLUEMultiTaskDataset(Dataset):
    """Multi-task dataset (kept for preprocessing compatibility)"""

    def __init__(self, task_names: List[str], split="train", max_len=128, tokenizer_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.task_names = task_names

        self.task_to_id = {task: i for i, task in enumerate(task_names)}
        self.data = []

        for task_name in task_names:
            task_data = self._load_task_data(task_name, split)
            self.data.extend(task_data)

    def _load_task_data(self, task_name: str, split: str) -> List[Dict]:
        if task_name not in GLUE_TASKS:
            raise ValueError(f"Task {task_name} not found in GLUE_TASKS")

        config = GLUE_TASKS[task_name]

        if task_name == "mnli" and split == "validation":
            # For MNLI validation, load both matched and mismatched
            dataset_matched = load_dataset("glue", task_name)["validation_matched"]
            dataset_mismatched = load_dataset("glue", task_name)["validation_mismatched"]
            dataset = list(dataset_matched) + list(dataset_mismatched)
        else:
            dataset = load_dataset("glue", task_name)[split]

        task_data = []

        for sample in dataset:
            text1, text2 = self._extract_texts(sample, task_name, config)

            if sample.get("label", -1) == -1:
                continue

            encoding = self._tokenize_texts(text1, text2)
            label = self._process_label(sample["label"], task_name)

            task_data.append({
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": label,
                "task_name": task_name,
                "task_id": self.task_to_id[task_name]
            })

        return task_data

    def _extract_texts(self, sample, task_name, config):
        if config["input_format"] == "single":
            if task_name == "cola":
                return sample["sentence"], None
            elif task_name == "sst2":
                return sample["sentence"], None
        else:
            if task_name in ["mrpc", "stsb"]:
                return sample["sentence1"], sample["sentence2"]
            elif task_name == "mnli":
                return sample["premise"], sample["hypothesis"]
            elif task_name == "rte":
                return sample["sentence1"], sample["sentence2"]
            elif task_name == "qnli":
                return sample["question"], sample["sentence"]
            elif task_name == "qqp":
                return sample["question1"], sample["question2"]
            elif task_name == "wnli":
                return sample["sentence1"], sample["sentence2"]

        return sample.get("sentence", ""), None

    def _tokenize_texts(self, text1, text2):
        if text2 is None:
            return self.tokenizer(
                text1,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )
        else:
            return self.tokenizer(
                text1, text2,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )

    def _process_label(self, label, task_name):
        if task_name == "stsb":
            return float(label) / 5.0
        return int(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def create_glue_dataloaders(task_names: List[str], batch_size: int = 128, max_len: int = 128):
    """Create multi-task dataloaders (updated to use validation instead of test)"""
    train_dataset = GLUEMultiTaskDataset(task_names, split="train", max_len=max_len)
    val_dataset = GLUEMultiTaskDataset(task_names, split="validation", max_len=max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_loader, val_loader


def collate_fn(batch):
    """Custom collate function for multi-task batches"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([torch.tensor(item["labels"]) for item in batch])
    task_names = [item["task_name"] for item in batch]
    task_ids = torch.tensor([item["task_id"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "task_names": task_names,
        "task_ids": task_ids
    }