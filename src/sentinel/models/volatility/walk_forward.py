"""Walk-forward evaluation: train on the past, predict the next slice, then roll forward.

This is the sound way to test on non-stationary market data. Instead of one fixed split it
returns a single long series of out-of-sample predictions spanning many years, which is also
what the downstream backtest needs.
"""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

from sentinel.features.technical import FEATURE_COLUMNS


def walk_forward(df, fit_predict, start="2016-01-01", embargo_days=5, feature_cols=FEATURE_COLUMNS,
                 min_train=1000):
    """Roll a yearly window forward. ``fit_predict(X_train, y_train, X_test) -> proba`` trains a
    fresh model on each fold. Returns date, ticker, label, and predicted probability per test row."""
    df = df.dropna(subset=list(feature_cols) + ["label_highvol"]).sort_values("date").copy()
    df["label_highvol"] = df["label_highvol"].astype(int)
    dates = pd.to_datetime(df["date"])
    bounds = pd.date_range(start=start, end=dates.max() + pd.offsets.YearBegin(), freq="YS")
    embargo = pd.Timedelta(days=embargo_days)

    folds = []
    for begin, end in zip(bounds[:-1], bounds[1:]):
        train = dates < (begin - embargo)
        test = (dates >= begin) & (dates < end)
        if train.sum() < min_train or test.sum() == 0:
            continue
        scaler = StandardScaler().fit(df.loc[train, feature_cols])
        x_train = scaler.transform(df.loc[train, feature_cols]).astype("float32")
        x_test = scaler.transform(df.loc[test, feature_cols]).astype("float32")
        y_train = df.loc[train, "label_highvol"].to_numpy().astype("float32")
        proba = fit_predict(x_train, y_train, x_test)
        block = df.loc[test, ["date", "ticker", "label_highvol"]].copy()
        block["proba"] = proba
        folds.append(block)

    return pd.concat(folds, ignore_index=True)


def xgb_fit_predict(x_train, y_train, x_test):
    from sentinel.models.volatility.xgb import train_xgb, xgb_proba
    return xgb_proba(train_xgb((x_train, y_train)), x_test)


def mlp_fit_predict(x_train, y_train, x_test):
    from sentinel.models.volatility.mlp import train_mlp, mlp_proba
    return mlp_proba(train_mlp((x_train, y_train), epochs=30), x_test)
