"""Exploratory data analysis for the volatility-regime dataset.

Generates figures in reports/figures/ and prints summary statistics that feed the Dataset Summary draft.
All analysis is framed around the hypothesis: can these features anticipate a high-volatility regime?
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel.features.technical import FEATURE_COLUMNS
from sentinel.eda import plots

FIG = ROOT / "reports" / "figures"


def main() -> None:
    df = pd.read_parquet(ROOT / "data" / "processed" / "features_labeled.parquet")
    labeled = df.dropna(subset=["label_highvol"]).copy()
    labeled["label_highvol"] = labeled["label_highvol"].astype(int)

    print("=" * 70)
    print(f"Rows total: {len(df):,}   labeled: {len(labeled):,}   tickers: {df['ticker'].nunique()}")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")

    # class balance
    counts = labeled["label_highvol"].value_counts().sort_index()
    print("\nClass balance:")
    for k, v in counts.items():
        print(f"  class {k}: {v:,} ({v / len(labeled):.1%})")
    print(f"  -> positives (high-vol): {counts.get(1, 0):,} "
          f"({'MEETS' if counts.get(1, 0) >= 10000 else 'below'} 10k-per-class target)")

    # per-ticker positives
    per = labeled.groupby("ticker")["label_highvol"].agg(["size", "mean"]).round(3)
    print("\nPer-ticker (rows, high-vol rate):")
    print(per.to_string())

    # missingness on features
    miss = df[FEATURE_COLUMNS].isna().mean().sort_values(ascending=False)
    print("\nFeature missingness (top 5, mostly warm-up windows):")
    print((miss.head(5) * 100).round(2).to_string())

    # figures
    plots.plot_class_balance(labeled["label_highvol"], out=FIG / "class_balance.png")
    plots.plot_correlation(labeled, FEATURE_COLUMNS, out=FIG / "feature_correlation.png")
    plots.plot_feature_by_label(labeled, FEATURE_COLUMNS, out=FIG / "feature_by_label.png")

    # stationarity
    adf = plots.adf_report(labeled, FEATURE_COLUMNS)
    print("\nADF stationarity test (p<0.05 => stationary):")
    print(adf.round(4).to_string(index=False))
    adf.to_csv(ROOT / "reports" / "adf_report.csv", index=False)

    # separability: mean feature value by class (quick signal check)
    sep = labeled.groupby("label_highvol")[FEATURE_COLUMNS].mean().T
    sep.columns = ["normal", "high_vol"]
    sep["abs_diff"] = (sep["high_vol"] - sep["normal"]).abs()
    print("\nMean feature value by class (sorted by separation):")
    print(sep.sort_values("abs_diff", ascending=False).round(4).to_string())

    print("\nFigures saved to reports/figures/:")
    for p in sorted(FIG.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
