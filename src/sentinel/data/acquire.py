"""Daily OHLCV download from Yahoo Finance, with Stooq as a backup."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]


def _tidy_from_yf(df, ticker):
    df = df.rename(columns=str.lower).reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]
    df["ticker"] = ticker
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df[COLUMNS]


def download_yfinance(tickers, start, end, interval="1d"):
    import yfinance as yf

    frames = []
    for t in tickers:
        raw = yf.download(t, start=start, end=end, interval=interval,
                          auto_adjust=False, progress=False, threads=False)
        if raw is None or raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        frames.append(_tidy_from_yf(raw, t))
    if not frames:
        raise RuntimeError("yfinance returned nothing for the requested tickers.")
    return pd.concat(frames, ignore_index=True)


def download_stooq(tickers, start, end):
    from pandas_datareader import data as pdr

    frames = []
    for t in tickers:
        raw = pdr.DataReader(t, "stooq", start=start, end=end)
        if raw is None or raw.empty:
            continue
        raw = raw.sort_index().rename(columns=str.lower).reset_index()
        raw["ticker"] = t
        raw["adj_close"] = raw["close"]  # Stooq has no adjusted close
        raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)
        frames.append(raw[COLUMNS])
    if not frames:
        raise RuntimeError("Stooq returned nothing for the requested tickers.")
    return pd.concat(frames, ignore_index=True)


def acquire(tickers, start, end, interval="1d", save_to=None):
    try:
        df = download_yfinance(tickers, start, end, interval)
    except Exception as exc:
        print(f"yfinance failed ({exc}); falling back to Stooq")
        df = download_stooq(tickers, start, end)

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_to, index=False)
    return df
