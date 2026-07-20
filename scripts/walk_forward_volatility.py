"""Walk-forward evaluation of the volatility circuit-breaker (MLP vs XGBoost).

    python scripts/walk_forward_volatility.py
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

import torch
torch.set_num_threads(1)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, ConfusionMatrixDisplay, confusion_matrix)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel.models.volatility.walk_forward import walk_forward, xgb_fit_predict, mlp_fit_predict
from sentinel.models.volatility.evaluate import pick_threshold_max_f1, scores

FIG = ROOT / "reports" / "figures"


def main():
    df = pd.read_parquet(ROOT / "data/processed/features_labeled.parquet")
    FIG.mkdir(parents=True, exist_ok=True)

    runs = {"XGBoost": xgb_fit_predict, "MLP": mlp_fit_predict}
    fig, ax = plt.subplots(figsize=(5, 4))

    for name, fp in runs.items():
        oos = walk_forward(df, fp, start="2016-01-01")
        y, proba = oos["label_highvol"].to_numpy(), oos["proba"].to_numpy()
        thr = pick_threshold_max_f1(y, proba)
        s = scores(y, proba, thr)
        print(f"\n{name}  (out-of-sample {oos['date'].min().date()} .. {oos['date'].max().date()}, "
              f"{len(oos):,} rows)")
        print(f"  PR-AUC {s['pr_auc']:.3f} | ROC-AUC {s['roc_auc']:.3f} | base rate {y.mean():.3f}")
        print(f"  max-F1 threshold {thr:.2f} -> precision {s['precision']:.3f} "
              f"recall {s['recall']:.3f} F1 {s['f1']:.3f}")

        oos["year"] = pd.to_datetime(oos["date"]).dt.year
        per_year = oos.groupby("year").apply(
            lambda g: average_precision_score(g["label_highvol"], g["proba"])
            if g["label_highvol"].nunique() > 1 else float("nan"), include_groups=False)
        print("  PR-AUC by year:", {int(k): round(v, 2) for k, v in per_year.items()})

        prec, rec, _ = precision_recall_curve(y, proba)
        ax.plot(rec, prec, label=f"{name} (AP={s['pr_auc']:.2f})")
        ConfusionMatrixDisplay(confusion_matrix(y, (proba >= thr).astype(int)),
                               display_labels=["Normal", "High-vol"]).plot(cmap="Blues", colorbar=False)
        plt.title(f"{name}: walk-forward confusion matrix")
        plt.savefig(FIG / f"volatility_wf_confusion_{name.lower()}.png", dpi=150, bbox_inches="tight")
        plt.close()

    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title("Volatility circuit-breaker: walk-forward precision-recall")
    ax.legend(fontsize=8)
    fig.savefig(FIG / "volatility_wf_pr_curve.png", dpi=150, bbox_inches="tight")
    print(f"\nfigures -> {FIG.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
