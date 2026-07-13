"""EDA on the processed dataset: prints summary stats and writes figures to reports/figures/."""
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

    print(f"{len(df):,} rows, {len(labeled):,} labeled, {df['ticker'].nunique()} tickers")
    print(f"dates {df['date'].min().date()} to {df['date'].max().date()}")

    counts = labeled["label_highvol"].value_counts().sort_index()
    for k, v in counts.items():
        print(f"  class {k}: {v:,} ({v / len(labeled):.1%})")

    per = labeled.groupby("ticker")["label_highvol"].agg(["size", "mean"]).round(3)
    print("\nper ticker (rows, high-vol rate):")
    print(per.to_string())

    miss = df[FEATURE_COLUMNS].isna().mean().sort_values(ascending=False)
    print("\ntop feature missingness (warm-up windows):")
    print((miss.head(5) * 100).round(2).to_string())

    plots.plot_class_balance(labeled["label_highvol"], out=FIG / "class_balance.png")
    plots.plot_correlation(labeled, FEATURE_COLUMNS, out=FIG / "feature_correlation.png")
    plots.plot_feature_by_label(labeled, FEATURE_COLUMNS, out=FIG / "feature_by_label.png")

    adf = plots.adf_report(labeled, FEATURE_COLUMNS)
    print("\nADF stationarity:")
    print(adf.round(4).to_string(index=False))
    adf.to_csv(ROOT / "reports" / "adf_report.csv", index=False)

    sep = labeled.groupby("label_highvol")[FEATURE_COLUMNS].mean().T
    sep.columns = ["normal", "high_vol"]
    sep["abs_diff"] = (sep["high_vol"] - sep["normal"]).abs()
    print("\nmean by class (sorted by separation):")
    print(sep.sort_values("abs_diff", ascending=False).round(4).to_string())

    print(f"\nfigures -> {FIG.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
