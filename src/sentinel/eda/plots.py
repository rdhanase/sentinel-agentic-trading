"""Plot helpers for the EDA notebook and script."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import adfuller


def _save(fig, out):
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig


def plot_class_balance(labels, out=None):
    counts = labels.dropna().astype(int).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.bar(["Normal (0)", "High-vol (1)"], counts.reindex([0, 1]).fillna(0), color=["#4472C4", "#C00000"])
    total = counts.sum()
    for i, v in enumerate(counts.reindex([0, 1]).fillna(0)):
        ax.text(i, v, f"{v:,.0f}\n({v / total:.1%})", ha="center", va="bottom", fontsize=9)
    ax.set_title("Volatility-regime class balance")
    ax.set_ylabel("count")
    return _save(fig, out)


def plot_correlation(df, feature_cols, out=None):
    corr = df[feature_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature correlation matrix")
    return _save(fig, out)


def plot_feature_by_label(df, feature_cols, label_col="label_highvol", out=None):
    d = df.dropna(subset=[label_col]).copy()
    d[label_col] = d[label_col].astype(int)
    n = len(feature_cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    for ax, col in zip(axes.ravel(), feature_cols):
        for lab, color in [(0, "#4472C4"), (1, "#C00000")]:
            sns.kdeplot(d.loc[d[label_col] == lab, col].dropna(), ax=ax, color=color,
                        fill=True, alpha=0.3, label=f"class {lab}")
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle("Feature distributions by volatility regime", y=1.01)
    fig.tight_layout()
    return _save(fig, out)


def adf_report(df, feature_cols):
    # ADF test per feature; low p-value means stationary
    rows = []
    for col in feature_cols:
        s = df[col].dropna()
        if len(s) < 50:
            continue
        stat, p, *_ = adfuller(s, autolag="AIC")
        rows.append({"feature": col, "adf_stat": stat, "p_value": p, "stationary_5pct": p < 0.05})
    return pd.DataFrame(rows)
