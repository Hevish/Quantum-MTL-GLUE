# main.py
import torch
from hybrid_model import MTLModel
from dataset import load_glue_task
from train import train_glue_baselines_style, evaluate
from tasks import GLUE_TASKS
import pennylane as qml


def get_quantum_params(tasks):
    tasks_qubits = []
    layers_per_task = []
    task_observables = []
    task_order = list(tasks.keys())

    for task_name, task_config in tasks.items():
        num_classes = task_config['num_classes']

        if task_name == 'stsb':  # Regression task
            n_qubits = 1
            n_layers = 1
            observables = [qml.PauliZ(0)]
        elif num_classes == 2:  # Binary classification
            n_qubits = 1
            n_layers = 1
            observables = [qml.PauliZ(0)]
        elif num_classes == 3:  # 3-class classification
            n_qubits = 2
            n_layers = 1
            observables = [qml.PauliZ(0), qml.PauliZ(1), qml.PauliX(0) @ qml.PauliX(1)]
        else:  # Multi-class (>3)
            n_qubits = 4
            n_layers = 2
            # Create observables for multi-class
            observables = ([qml.PauliZ(i) for i in range(4)] +
                           [qml.PauliZ(i) @ qml.PauliZ((i + 1) % 4) for i in range(4)])
            observables = observables[:num_classes]  # Take only what we need

        tasks_qubits.append(n_qubits)
        layers_per_task.append(n_layers)
        task_observables.append(observables)

    return {
        'tasks_qubits': tasks_qubits,
        'layers_per_task': layers_per_task,
        'task_observables': task_observables,
        'encoding_layers': 3,  # Fixed at 3 layers
        'task_order': task_order
    }

def main(USE_QUANTUM=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    task_names = ["cola", "sst2", "mrpc", "stsb", "mnli", "qnli", "qqp", "rte", "wnli"]
    # task_names = ["cola", "stsb", "wnli"]

    # Load individual task dataloaders
    print("Loading individual task dataloaders...")
    dataloaders = {}
    for task in task_names:
        print(f"Loading {task}...")
        dataloaders[task] = load_glue_task(task, batch_size=128)

    # Initialize model
    selected_tasks = {k: v for k, v in GLUE_TASKS.items() if k in task_names}
    if USE_QUANTUM:
        quantum_params = get_quantum_params(selected_tasks)
        model = MTLModel(vocab_size=30522, tasks=selected_tasks, QUANTUM=True, quantum_params=quantum_params).to(device)
    else:
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


    import json
    import os
    from datetime import datetime


    # Save results to JSON file
    results_to_save = {
        "timestamp": datetime.now().isoformat(),
        "model_type": "quantum" if USE_QUANTUM else "classical",
        "task_results": final_results,
        "macro_average": macro_avg,
        "tasks_evaluated": task_names,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "device_used": str(device)
    }

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Generate filename with timestamp and model type
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = "quantum" if USE_QUANTUM else "classical"
    filename = f"results/glue_results_{model_type}_{timestamp_str}.json"

    # Save to JSON file
    with open(filename, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\nResults saved to: {filename}")

if __name__ == "__main__":
    main()