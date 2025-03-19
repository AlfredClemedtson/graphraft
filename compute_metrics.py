from collections.abc import Callable

def compute_metrics(predss: list[list[int]], labelss: list[list[int]], metrics: list[str], do_print=True) -> dict[str, float]:
    results = {}
    for metric in metrics:
        match metric.lower().split('@'):
            case ["recall"]:
                result = _macro_average(_recall, predss, labelss)
            case ["recall", k]:
                result = _macro_average(_recall, predss, labelss, eval(k))
            case ["precision"]:
                result = _macro_average(_precision, predss, labelss)
            case ["f1"]:
                result = _macro_average(_f1, predss, labelss)
            case "hit", k:
                result = _macro_average(_hit, predss, labelss, eval(k))
            case ["mrr"]:
                result = _macro_average(_mrr, predss, labelss)
            case ["num_nodes"]:
                result = _macro_average(_cnt, predss, labelss)
            case _:
                print(f"{metric} is not a valid metric")
                result = float("nan")
        results[metric] = result
    max_length = max([len(metric) for metric in metrics])
    if do_print:
        for metric, value in results.items():
            print(f"{metric.ljust(max_length)}: {value:.3f}")
    return results

def _cnt(preds: list[int], _):
    return len(preds)

def _hits(preds: list[int], labels: list[int]) -> int:
    return len(set(preds).intersection(labels))

def _hit(preds: list[int], labels: list[int]) -> int:
    return 1 if _hits(preds, labels) > 0 else 0

def _precision(preds: list[int], labels: list[int]) -> float:
    return _hits(preds, labels) / len(preds) if len(preds) > 0 else 0

def _recall(preds: list[int], labels: list[int]) -> float:
    return _hits(preds, labels) / len(labels)

def _f1(preds: list[int], labels: list[int]) -> float:
    precision = _precision(preds, labels)
    rec = _recall(preds, labels)
    return 2 * precision * rec / (precision + rec) if precision != 0 else 0

def _best_rank(preds: list[int], labels: list[int]) -> float:
    # Rank of highest ranking label
    return min([preds.index(label)+1.0 if label in preds else float('inf') for label in labels])

def _mrr(preds: list[int], labels: list[int]) -> float:
    return 1/_best_rank(preds, labels)

def top_k(xs: list[int], k: int):
    k_ = len(xs) if k == -1 or len(xs) < k else k
    return xs[:k_]

def _macro_average(metric_func: Callable[[list[int], list[int]], float], predss, labelss, k=-1):
    ys = [metric_func(top_k(preds, k), labels) for preds, labels in zip(predss, labelss)]
    return sum(ys)/len(ys)