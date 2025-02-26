import numpy as np
import torch

def compute_metrics(preds: torch.tensor, labels: torch.tensor, metrics: list[str]) -> dict[str, float]:
    #preds: num_question * num_predictions
    results = {}
    for metric in metrics:
        match metric:
            case "recall":
                recalls = [len(set(predicted_nodes.to_list()).intersection(true_nodes.to_list())) for predicted_nodes, true_nodes in zip(preds, labels)]
                result = np.mean(recalls)
            case _:
                result = float("nan")
        results[metric] = result
