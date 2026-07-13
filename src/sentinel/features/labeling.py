"""Volatility-regime target and time-ordered splits.

The label is 1 when the next H days are more volatile than the trailing q-th percentile of volatility.
The threshold is built from past data only, so the future never leaks into the label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_realized_vol(ret, horizon):
    # value at t covers t+1..t+H; looks ahead, so it's only ever used as the label
    sq = ret.pow(2)
    return np.sqrt(sq.rolling(horizon).sum().shift(-horizon))


def trailing_percentile(series, lookback, q):
    # percentile over a window that ends the day before t
    return series.shift(1).rolling(lookback, min_periods=lookback // 2).quantile(q)


def make_labels(df, horizon, lookback, quantile):
    df = df.sort_values("date").copy()
    ret = df["ret_1"]
    df["fwd_rvol"] = forward_realized_vol(ret, horizon)
    base_rvol = ret.pow(2).rolling(horizon).sum().pow(0.5)
    df["vol_threshold"] = trailing_percentile(base_rvol, lookback, quantile)
    df["label_highvol"] = (df["fwd_rvol"] > df["vol_threshold"]).astype("Int64")
    df.loc[df["fwd_rvol"].isna() | df["vol_threshold"].isna(), "label_highvol"] = pd.NA
    return df


def time_split(dates, train_end, valid_end, embargo_days):
    # chronological split; the embargo gap stops a label window from straddling a boundary
    d = pd.to_datetime(dates)
    emb = pd.Timedelta(days=embargo_days)
    train = d <= pd.Timestamp(train_end)
    valid = (d > pd.Timestamp(train_end) + emb) & (d <= pd.Timestamp(valid_end))
    test = d > pd.Timestamp(valid_end) + emb
    return train, valid, test
