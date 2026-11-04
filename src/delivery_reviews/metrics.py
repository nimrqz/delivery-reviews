"""Métricas: F1 micro/macro por tarefa e por label, matriz de confusão."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from .labels import PROBLEMS, SENTIMENTS, URGENCIES


def problems_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    """F1 micro/macro e por label para a tarefa multi-label (matrizes multi-hot)."""
    per = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_per_label": {n: float(v) for n, v in zip(PROBLEMS, per)},
    }


def single_label_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], names: Sequence[str]
) -> dict[str, object]:
    """F1 micro/macro, por classe e matriz de confusão (sentimento ou urgência)."""
    labels = list(range(len(names)))
    per = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_per_label": {n: float(v) for n, v in zip(names, per)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


TASKS = {"sentimento": SENTIMENTS, "urgencia": URGENCIES}
