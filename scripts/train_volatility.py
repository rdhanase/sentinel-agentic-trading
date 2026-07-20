"""Train the volatility circuit-breaker (MLP) against an XGBoost baseline and compare them.

    python scripts/train_volatility.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel.models.volatility.data import prepare_tabular
from sentinel.models.volatility.mlp import train_mlp, mlp_proba
from sentinel.models.volatility.xgb import train_xgb, xgb_proba
from sentinel.models.volatility import evaluate

FIG = ROOT / "reports" / "figures"


def main():
    data = prepare_tabular(processed_path=ROOT / "data/processed/features_labeled.parquet",
                           config_path=str(ROOT / "config/config.yaml"))
    for split in ("train", "valid", "test"):
        X, y = data[split]
        print(f"{split:5} rows={len(y):>6}  high-vol={y.mean():.1%}")

    xgb = train_xgb(data["train"])
    mlp = train_mlp(data["train"], data["valid"])

    yva, yte = data["valid"][1], data["test"][1]
    probs = {
        "XGBoost": (xgb_proba(xgb, data["valid"][0]), xgb_proba(xgb, data["test"][0])),
        "MLP": (mlp_proba(mlp, data["valid"][0]), mlp_proba(mlp, data["test"][0])),
    }

    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, (vprob, tprob) in probs.items():
        thr = evaluate.pick_threshold(yva, vprob, target_precision=0.7)
        s = evaluate.scores(yte, tprob, thr)
        print(f"\n{name}: PR-AUC {s['pr_auc']:.3f} | ROC-AUC {s['roc_auc']:.3f} | "
              f"threshold {thr:.2f} -> precision {s['precision']:.3f} "
              f"recall {s['recall']:.3f} F1 {s['f1']:.3f}")
        print("  confusion [[TN FP][FN TP]]:", s["confusion"].tolist())
        prec, rec, _ = precision_recall_curve(yte, tprob)
        ax.plot(rec, prec, label=f"{name} (AP={s['pr_auc']:.2f})")
        ConfusionMatrixDisplay(s["confusion"], display_labels=["Normal", "High-vol"]).plot(
            cmap="Blues", colorbar=False)
        plt.title(f"{name}: test confusion matrix")
        plt.savefig(FIG / f"volatility_confusion_{name.lower()}.png", dpi=150, bbox_inches="tight")
        plt.close()

    base_rate = yte.mean()
    ax.axhline(base_rate, ls="--", color="gray", lw=1, label=f"base rate ({base_rate:.2f})")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title("Volatility circuit-breaker: precision-recall (test)")
    ax.legend(fontsize=8)
    fig.savefig(FIG / "volatility_pr_curve.png", dpi=150, bbox_inches="tight")
    print(f"\nfigures -> {FIG.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
