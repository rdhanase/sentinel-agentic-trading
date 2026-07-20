"""Prepare the volatility dataset: chronological train/valid/test splits with train-only scaling."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from sentinel.features.technical import FEATURE_COLUMNS
from sentinel.features.labeling import time_split
from sentinel.utils.config import load_config


def prepare_tabular(processed_path=None, config_path="config/config.yaml"):
    cfg = load_config(config_path)
    if processed_path is None:
        processed_path = Path(cfg.data.processed_dir) / "features_labeled.parquet"

    df = pd.read_parquet(processed_path)
    df = df.dropna(subset=FEATURE_COLUMNS + ["label_highvol"]).sort_values(["date", "ticker"])
    df["label_highvol"] = df["label_highvol"].astype(int)

    train, valid, test = time_split(df["date"], cfg.split.train_end,
                                    cfg.split.valid_end, cfg.split.embargo_days)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(df.loc[train, FEATURE_COLUMNS])  # fit on train only

    def xy(mask):
        X = scaler.transform(df.loc[mask, FEATURE_COLUMNS]).astype("float32")
        y = df.loc[mask, "label_highvol"].to_numpy().astype("float32")
        return X, y

    return {"features": FEATURE_COLUMNS, "scaler": scaler,
            "train": xy(train), "valid": xy(valid), "test": xy(test)}
