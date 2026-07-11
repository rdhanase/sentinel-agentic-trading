"""End-to-end dataset build: acquire -> clean -> engineer features -> label -> save.

Usage:
    python scripts/build_dataset.py --config config/config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# make the package importable when run as a script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel.utils.config import load_config
from sentinel.data.acquire import acquire
from sentinel.data.clean import clean_panel
from sentinel.features.technical import build_features, FEATURE_COLUMNS
from sentinel.features.labeling import make_labels


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    raw_path = ROOT / cfg.data.raw_dir / "ohlcv_raw.parquet"
    interim_path = ROOT / cfg.data.interim_dir / "ohlcv_clean.parquet"
    processed_path = ROOT / cfg.data.processed_dir / "features_labeled.parquet"
    report_path = ROOT / cfg.data.interim_dir / "cleaning_report.csv"

    print(f"[1/4] Acquiring {len(cfg.data.tickers)} tickers "
          f"({cfg.data.start} -> {cfg.data.end}) ...")
    raw = acquire(cfg.data.tickers, cfg.data.start, cfg.data.end, cfg.data.interval, save_to=raw_path)
    print(f"      raw rows: {len(raw):,}")

    print("[2/4] Cleaning ...")
    clean, report = clean_panel(raw)
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(interim_path, index=False)
    report.to_csv(report_path, index=False)
    print(report.to_string(index=False))

    print("[3/4] Engineering features + labeling per ticker ...")
    out = []
    for ticker, g in clean.groupby("ticker", sort=False):
        feats = build_features(g, cfg.features)
        labeled = make_labels(feats, cfg.label.horizon, cfg.label.lookback, cfg.label.quantile)
        out.append(labeled)
    dataset = pd.concat(out, ignore_index=True)

    print("[4/4] Saving processed dataset ...")
    keep = ["date", "ticker", "close", "adj_close", "volume"] + FEATURE_COLUMNS + \
           ["fwd_rvol", "vol_threshold", "label_highvol"]
    dataset = dataset[keep]
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(processed_path, index=False)

    labeled_rows = dataset["label_highvol"].notna().sum()
    pos = dataset["label_highvol"].dropna().astype(int).mean()
    print(f"\nDone. Processed rows: {len(dataset):,} | labeled: {labeled_rows:,} "
          f"| high-vol rate: {pos:.1%}")
    print(f"Saved -> {processed_path.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    main(ap.parse_args().config)
