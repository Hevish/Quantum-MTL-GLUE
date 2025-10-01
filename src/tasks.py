# tasks.py
GLUE_TASKS = {
    "cola": {
        "num_classes": 2,
        "metric": "matthews_correlation",
        "input_format": "single",
        "task_type": "classification"
    },
    "sst2": {
        "num_classes": 2,
        "metric": "accuracy",
        "input_format": "single",
        "task_type": "classification"
    },
    "mrpc": {
        "num_classes": 2,
        "metric": "f1",
        "input_format": "pair",
        "task_type": "classification"
    },
    "stsb": {
        "num_classes": 1,
        "metric": "spearmanr",
        "input_format": "pair",
        "task_type": "regression"
    },
    "mnli": {
        "num_classes": 3,
        "metric": "accuracy",
        "input_format": "pair",
        "task_type": "classification"
    },
    "qnli": {
        "num_classes": 2,
        "metric": "accuracy",
        "input_format": "pair",
        "task_type": "classification"
    },
    "qqp": {
        "num_classes": 2,
        "metric": "f1",
        "input_format": "pair",
        "task_type": "classification"
    },
    "rte": {
        "num_classes": 2,
        "metric": "accuracy",
        "input_format": "pair",
        "task_type": "classification"
    },
    "wnli": {  # Added WNLI task
        "num_classes": 2,
        "metric": "accuracy",
        "input_format": "pair",
        "task_type": "classification"
    }
}