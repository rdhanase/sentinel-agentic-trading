"""OHLCV data-quality fixes. Returns the cleaned panel plus a per-ticker note of what was wrong."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

PRICE_COLS = ["open", "high", "low", "close", "adj_close"]


def clean_ticker(df: pd.DataFrame, max_ffill: int = 3) -> Tuple[pd.DataFrame, Dict]:
    report: Dict[str, int] = {}
    df = df.sort_values("date").copy()

    # repeated dates -> keep the latest
    report["duplicate_dates"] = int(df["date"].duplicated().sum())
    df = df.drop_duplicates(subset="date", keep="last")

    # zero/negative prices aren't real; null them so they get repaired below
    bad_price = (df[PRICE_COLS] <= 0).any(axis=1)
    report["nonpositive_prices"] = int(bad_price.sum())
    df.loc[bad_price, PRICE_COLS] = np.nan

    # occasionally high < low; rebuild the bounds from the four prices
    report["inverted_high_low"] = int((df["high"] < df["low"]).sum())
    df["high"] = df[["high", "low", "open", "close"]].max(axis=1)
    df["low"] = df[["high", "low", "open", "close"]].min(axis=1)

    # carry prices over short gaps; missing volume means no trading
    report["missing_close_before"] = int(df["close"].isna().sum())
    df[PRICE_COLS] = df[PRICE_COLS].ffill(limit=max_ffill)
    df["volume"] = df["volume"].fillna(0)

    # still no close after the fill -> drop it
    report["rows_dropped_missing_close"] = int(df["close"].isna().sum())
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    report["n_rows_final"] = int(len(df))
    return df, report


def clean_panel(df: pd.DataFrame, max_ffill: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cleaned, reports = [], []
    for ticker, g in df.groupby("ticker", sort=False):
        cg, rep = clean_ticker(g, max_ffill=max_ffill)
        cleaned.append(cg)
        reports.append({"ticker": ticker, **rep})
    panel = pd.concat(cleaned, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
    return panel, pd.DataFrame(reports)
