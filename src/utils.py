# utils.py
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import spearmanr
from tasks import GLUE_TASKS


def compute_metrics(predictions, labels, task_name):
    """Compute task-specific metrics"""
    predictions = np.array(predictions)
    labels = np.array(labels)

    task_config = GLUE_TASKS[task_name]
    metric_name = task_config["metric"]

    if metric_name == "accuracy":
        return {"accuracy": accuracy_score(labels, predictions)}

    elif metric_name == "f1":
        return {
            "f1": f1_score(labels, predictions, average="binary" if task_config["num_classes"] == 2 else "macro"),
            "accuracy": accuracy_score(labels, predictions)
        }

    elif metric_name == "matthews_correlation":
        return {
            "matthews_correlation": matthews_corrcoef(labels, predictions),
            "accuracy": accuracy_score(labels, predictions)
        }

    elif metric_name == "spearmanr":
        correlation, p_value = spearmanr(labels, predictions)
        return {
            "spearmanr": correlation,
            "p_value": p_value
        }

    else:
        return {"accuracy": accuracy_score(labels, predictions)}