"""Volatility-regime labeling and leakage-safe time splits.

Target: y_t = 1 (high-volatility / high-risk) when the *forward* realized volatility over the next H days
exceeds the q-th percentile of realized volatility computed from a *trailing* window (past data only).
Using a trailing percentile keeps the class balance roughly stationary across market regimes and, crucially,
never lets future information set the threshold.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def forward_realized_vol(ret: pd.Series, horizon: int) -> pd.Series:
    """Realized volatility over the next ``horizon`` bars (uses future returns; this is a LABEL only).

    The value at time t summarizes returns over t+1 .. t+H:
    ``rolling(H).sum()`` at index t+H covers t+1..t+H, and ``shift(-H)`` moves it back to index t.
    """
    sq = ret.pow(2)
    fwd_sum = sq.rolling(horizon).sum().shift(-horizon)
    return np.sqrt(fwd_sum)


def trailing_percentile(series: pd.Series, lookback: int, q: float) -> pd.Series:
    """q-th percentile over a trailing window ending *before* the current bar (no look-ahead)."""
    return series.shift(1).rolling(lookback, min_periods=lookback // 2).quantile(q)


def make_labels(df: pd.DataFrame, horizon: int, lookback: int, quantile: float) -> pd.DataFrame:
    """Attach forward realized vol, the trailing threshold, and the binary high-vol label."""
    df = df.sort_values("date").copy()
    ret = df["ret_1"]
    df["fwd_rvol"] = forward_realized_vol(ret, horizon)
    # threshold built from the *realized* (contemporaneous) vol distribution, trailing only
    base_rvol = ret.pow(2).rolling(horizon).sum().pow(0.5)
    df["vol_threshold"] = trailing_percentile(base_rvol, lookback, quantile)
    df["label_highvol"] = (df["fwd_rvol"] > df["vol_threshold"]).astype("Int64")
    df.loc[df["fwd_rvol"].isna() | df["vol_threshold"].isna(), "label_highvol"] = pd.NA
    return df


def time_split(dates: pd.Series, train_end: str, valid_end: str, embargo_days: int
               ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Chronological train/valid/test masks with an embargo gap after each boundary.

    The embargo drops rows whose forward-looking label window would otherwise overlap the next split,
    which is the standard purge/embargo guard for time-series cross-validation.
    """
    d = pd.to_datetime(dates)
    train_end_ts = pd.Timestamp(train_end)
    valid_end_ts = pd.Timestamp(valid_end)
    emb = pd.Timedelta(days=embargo_days)

    train = d <= train_end_ts
    valid = (d > train_end_ts + emb) & (d <= valid_end_ts)
    test = d > valid_end_ts + emb
    return train, valid, test
