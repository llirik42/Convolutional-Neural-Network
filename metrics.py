import numpy as np


__all__ = ["print_metrics"]


def print_metrics(tp: np.ndarray, fp: np.ndarray, tn: np.ndarray, fn: np.ndarray, support: np.ndarray) -> None:
    def get_formatted_float(value: float) -> str:
        return f"{round(value, 2):.2f}"

    precision: np.ndarray = _calculate_precision(tp=tp, fp=fp)
    recall: np.ndarray = _calculate_recall(tp=tp, fn=fn)
    f1_score: np.ndarray = __calculate_f_score(recall=recall, precision=precision, beta=1)

    print(f"{'Accuracy':20s}{get_formatted_float(_calculate_accuracy(tp=tp, fp=fp, tn=tn, fn=fn))}")
    print(f"{'Macro precision':20s}{get_formatted_float(_calculate_macro_average_precision(precision=precision))}")
    print(f"{'Macro recall':20s}{get_formatted_float(_calculate_macro_average_recall(recall=recall))}")
    print(f"{'Micro precision':20s}{get_formatted_float(_calculate_micro_average_precision(tp=tp, fp=fp))}")
    print(f"{'Micro recall':20s}{get_formatted_float(_calculate_micro_average_recall(tp=tp, fn=fn))}")
    print()

    print(f"{' ':11s}Precision{' ':4s}Recall{' ':4s}F1-Score{' ':4s}Support")
    for i in range(tp.shape[0]):
        print(
            f"Class {i}{' ':4s}{get_formatted_float(precision[i])}{' ':9s}{get_formatted_float(recall[i])}{' ':6s}{get_formatted_float(f1_score[i])}{' ':8s}{support[i]}"
        )


def _calculate_accuracy(tp: np.ndarray, fp: np.ndarray, tn: np.ndarray, fn: np.ndarray) -> float:
    total_tp: float = tp.sum()
    total_fp: float = fp.sum()
    total_tn: float = tn.sum()
    total_fn: float = fn.sum()

    return (total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn)


def _calculate_precision(tp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    length: int = tp.shape[0]
    result: np.ndarray = np.empty(length)

    for i in range(length):
        result[i] = tp[i] / (tp[i] + fp[i])

    return result


def _calculate_recall(tp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    length: int = tp.shape[0]
    result: np.ndarray = np.empty(length)

    for i in range(length):
        result[i] = tp[i] / (tp[i] + fn[i])

    return result


def _calculate_macro_average_precision(precision: np.ndarray) -> float:
    return precision.mean()


def _calculate_macro_average_recall(recall: np.ndarray) -> float:
    return recall.mean()


def _calculate_micro_average_precision(tp: np.ndarray, fp: np.ndarray) -> float:
    tps: float = sum(tp)
    fps: float = sum(fp)

    return tps / (tps + fps)


def _calculate_micro_average_recall(tp: np.ndarray, fn: np.ndarray) -> float:
    tps: float = sum(tp)
    fns: float = sum(fn)

    return tps / (tps + fns)


def __calculate_f_score(recall: np.ndarray, precision: np.ndarray, beta: float) -> np.ndarray:
    beta_squared: float = beta**2
    return (1 + beta_squared) * precision * recall / (beta_squared * precision + recall)
