"""Metrics for the imbalanced volatility task, plus a precision-targeted threshold."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             precision_score, recall_score, f1_score, confusion_matrix)


def pick_threshold(y, proba, target_precision=0.7):
    """Highest-recall threshold that still meets the target precision (tuned on validation).

    A false halt is costly, so we want to be confident before flagging high risk.
    """
    prec, rec, thr = precision_recall_curve(y, proba)
    ok = prec[:-1] >= target_precision
    if not ok.any():
        return 0.5
    return float(thr[np.argmax(rec[:-1] * ok)])


def pick_threshold_max_f1(y, proba):
    """Threshold that maximizes F1 on the positive class."""
    prec, rec, thr = precision_recall_curve(y, proba)
    if len(thr) == 0:
        return 0.5
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thr[np.argmax(f1)])


def scores(y, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {"pr_auc": average_precision_score(y, proba),
            "roc_auc": roc_auc_score(y, proba),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "threshold": threshold,
            "confusion": confusion_matrix(y, pred)}
