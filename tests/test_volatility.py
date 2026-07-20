import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.models.volatility.evaluate import pick_threshold, scores


def test_pick_threshold_meets_target_precision():
    y = np.array([0, 0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
    thr = pick_threshold(y, proba, target_precision=1.0)
    assert precision_score(y, (proba >= thr).astype(int)) == 1.0


def test_scores_has_expected_metrics():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.8, 0.3, 0.6])
    s = scores(y, p, 0.5)
    assert {"pr_auc", "roc_auc", "precision", "recall", "f1"} <= set(s)
    assert s["confusion"].shape == (2, 2)


def test_focal_loss_rewards_correct_confidence():
    torch = pytest.importorskip("torch")
    from sentinel.models.volatility.losses import focal_loss
    logits = torch.tensor([3.0, -3.0])
    targets = torch.tensor([1.0, 0.0])
    good = focal_loss(logits, targets)
    bad = focal_loss(-logits, targets)
    assert good.ndim == 0 and good.item() < bad.item()


def test_mlp_output_shape():
    torch = pytest.importorskip("torch")
    from sentinel.models.volatility.mlp import MLP
    model = MLP(15)
    assert model(torch.randn(8, 15)).shape == (8,)
