"""Stationary technical features engineered from OHLCV data (pure pandas, no TA-Lib).

Raw prices are non-stationary, so every feature here is a return, a bounded oscillator, a ratio, or a
z-score. These become the inputs to the volatility circuit-breaker.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def log_return(close: pd.Series, periods: int = 1) -> pd.Series:
    return np.log(close / close.shift(periods))


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index, scaled to [0, 1]."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    return (out.fillna(50.0)) / 100.0


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line, signal, and histogram, each normalized by price to be scale-free across tickers."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return pd.DataFrame({
        "macd_line": line / close,
        "macd_signal": sig / close,
        "macd_hist": hist / close,
    })


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range normalized by close (fraction of price)."""
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_abs = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr_abs / close


def bollinger_width(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """Bollinger Band width = (upper - lower) / middle. Already a stationary ratio."""
    mid = close.rolling(window).mean()
    sd = close.rolling(window).std()
    return (2 * num_std * sd) / mid


def realized_vol(ret: pd.Series, window: int) -> pd.Series:
    """Rolling realized volatility (std of log returns) over ``window`` days."""
    return ret.rolling(window).std()


def volume_zscore(volume: pd.Series, window: int = 21) -> pd.Series:
    mean = volume.rolling(window).mean()
    std = volume.rolling(window).std()
    return (volume - mean) / std.replace(0.0, np.nan)


def build_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Engineer all features for a single ticker's cleaned OHLCV frame."""
    df = df.sort_values("date").copy()
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]

    df["ret_1"] = log_return(close, 1)
    for h in cfg.return_horizons:
        df[f"ret_{h}"] = log_return(close, h)
    for w in cfg.vol_windows:
        df[f"rvol_{w}"] = realized_vol(df["ret_1"], w)

    df["rsi"] = rsi(close, cfg.rsi_period)
    df = pd.concat([df, macd(close, *cfg.macd)], axis=1)
    df["atr"] = atr(high, low, close, cfg.atr_period)
    df["bb_width"] = bollinger_width(close, int(cfg.bollinger[0]), float(cfg.bollinger[1]))
    df["vol_z"] = volume_zscore(vol, cfg.volume_zscore_window)
    df["hl_range"] = (high - low) / close
    return df


FEATURE_COLUMNS = [
    "ret_1", "ret_5", "ret_10", "ret_20",
    "rvol_10", "rvol_21", "rvol_63",
    "rsi", "macd_line", "macd_signal", "macd_hist",
    "atr", "bb_width", "vol_z", "hl_range",
]
