"""Technical indicators used as features. All returns/ratios/z-scores so they stay comparable
across tickers and over time."""
from __future__ import annotations

import numpy as np
import pandas as pd


def log_return(close, periods=1):
    return np.log(close / close.shift(periods))


def rsi(close, period=14):
    # Wilder's RSI, rescaled to 0-1
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0) / 100.0


def macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    # divided by price so it's on the same scale for every ticker
    return pd.DataFrame({"macd_line": line / close, "macd_signal": sig / close, "macd_hist": hist / close})


def atr(high, low, close, period=14):
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / close


def bollinger_width(close, window=20, num_std=2.0):
    mid = close.rolling(window).mean()
    sd = close.rolling(window).std()
    return (2 * num_std * sd) / mid


def realized_vol(ret, window):
    return ret.rolling(window).std()


def volume_zscore(volume, window=21):
    mean = volume.rolling(window).mean()
    std = volume.rolling(window).std()
    return (volume - mean) / std.replace(0.0, np.nan)


def build_features(df, cfg):
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
