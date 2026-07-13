"""Build the modeling dataset: download, clean, add features, label, and save.

    python scripts/build_dataset.py --config config/config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

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

    print(f"Downloading {len(cfg.data.tickers)} tickers ({cfg.data.start} to {cfg.data.end})")
    raw = acquire(cfg.data.tickers, cfg.data.start, cfg.data.end, cfg.data.interval, save_to=raw_path)
    print(f"  {len(raw):,} raw rows")

    clean, report = clean_panel(raw)
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(interim_path, index=False)
    report.to_csv(report_path, index=False)
    print(report.to_string(index=False))

    out = []
    for ticker, g in clean.groupby("ticker", sort=False):
        feats = build_features(g, cfg.features)
        out.append(make_labels(feats, cfg.label.horizon, cfg.label.lookback, cfg.label.quantile))
    dataset = pd.concat(out, ignore_index=True)

    keep = ["date", "ticker", "close", "adj_close", "volume"] + FEATURE_COLUMNS + \
           ["fwd_rvol", "vol_threshold", "label_highvol"]
    dataset = dataset[keep]
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(processed_path, index=False)

    labeled = dataset["label_highvol"].notna().sum()
    pos = dataset["label_highvol"].dropna().astype(int).mean()
    print(f"\n{len(dataset):,} rows, {labeled:,} labeled, {pos:.1%} high-vol")
    print(f"saved -> {processed_path.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    main(ap.parse_args().config)
