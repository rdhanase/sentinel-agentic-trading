"""Identify and correct common data-quality issues in raw OHLCV data.

Every correction is recorded in a report dict so the Dataset Summary can describe exactly what was
found and how it was handled.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

PRICE_COLS = ["open", "high", "low", "close", "adj_close"]


def clean_ticker(df: pd.DataFrame, max_ffill: int = 3) -> Tuple[pd.DataFrame, Dict]:
    """Clean a single ticker's OHLCV frame. Returns (clean_df, issue_report)."""
    report: Dict[str, int] = {}
    df = df.sort_values("date").copy()

    # 1. Duplicate trading dates -> keep the last observation.
    dup = df["date"].duplicated().sum()
    report["duplicate_dates"] = int(dup)
    df = df.drop_duplicates(subset="date", keep="last")

    # 2. Non-positive or impossible prices -> mark as missing for repair.
    bad_price = (df[PRICE_COLS] <= 0).any(axis=1)
    report["nonpositive_prices"] = int(bad_price.sum())
    df.loc[bad_price, PRICE_COLS] = np.nan

    # 3. Inverted bars (high < low) or close outside [low, high] -> flag and repair bounds.
    inverted = (df["high"] < df["low"]).sum()
    report["inverted_high_low"] = int(inverted)
    df["high"] = df[["high", "low", "open", "close"]].max(axis=1)
    df["low"] = df[["high", "low", "open", "close"]].min(axis=1)

    # 4. Missing values -> limited forward-fill (prices persist), volume gaps -> 0.
    report["missing_close_before"] = int(df["close"].isna().sum())
    df[PRICE_COLS] = df[PRICE_COLS].ffill(limit=max_ffill)
    df["volume"] = df["volume"].fillna(0)

    # 5. Rows still missing a close after ffill are unusable -> drop.
    remaining = df["close"].isna().sum()
    report["rows_dropped_missing_close"] = int(remaining)
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    report["n_rows_final"] = int(len(df))
    return df, report


def clean_panel(df: pd.DataFrame, max_ffill: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Clean every ticker in a long OHLCV panel.

    Returns the cleaned panel and a per-ticker issue report as a DataFrame.
    """
    cleaned, reports = [], []
    for ticker, g in df.groupby("ticker", sort=False):
        cg, rep = clean_ticker(g, max_ffill=max_ffill)
        rep = {"ticker": ticker, **rep}
        cleaned.append(cg)
        reports.append(rep)
    panel = pd.concat(cleaned, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
    return panel, pd.DataFrame(reports)
