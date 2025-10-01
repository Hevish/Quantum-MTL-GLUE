import pickle
import os
from dataset import GLUEMultiTaskDataset


def preprocess_and_cache():
    cache_file = "glue_processed_data.pkl"

    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print("Processing GLUE data for the first time...")
    # Updated task list to include all 9 GLUE tasks including WNLI
    task_names = ["cola", "sst2", "mrpc", "stsb", "mnli", "qnli", "rte", "wnli"]

    # Process train and validation
    train_data = GLUEMultiTaskDataset(task_names, split="train")
    test_data = GLUEMultiTaskDataset(task_names, split="test")

    cached_data = {
        "train": train_data.data,
        "test": test_data.data,
        "task_to_id": train_data.task_to_id
    }

    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)

    print(f"Data cached to {cache_file}")
    return cached_data


if __name__ == "__main__":
    preprocess_and_cache()