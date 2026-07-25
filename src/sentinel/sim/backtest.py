"""Offline paper-trading backtest for the volatility circuit-breaker.

The gate turns model risk probabilities into a position multiplier: trade normally when calm,
scale down when risk is elevated, halt when it's high. Everything here is deliberately simple
and depends only on pandas/numpy so it runs anywhere the model doesn't.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def risk_mode(proba, t_scale=0.4, t_halt=0.7):
    if proba >= t_halt:
        return "HALT"
    if proba >= t_scale:
        return "SCALED"
    return "NORMAL"


def gate_multiplier(proba, t_scale=0.4, t_halt=0.7):
    """Fraction of the base position to keep given the model's high-vol probability."""
    return {"NORMAL": 1.0, "SCALED": 0.5, "HALT": 0.0}[risk_mode(proba, t_scale, t_halt)]


def build_positions(oos_df, gate_on: bool, base=1.0, t_scale=0.4, t_halt=0.7):
    """Base is an equal-weight long book (position ``base`` per ticker-day). With the gate on
    each position is scaled by the circuit-breaker; with it off the book is untouched."""
    pos = oos_df[["date", "ticker"]].copy()
    if gate_on:
        mult = oos_df["proba"].map(lambda p: gate_multiplier(p, t_scale, t_halt))
        pos["position"] = base * mult.to_numpy()
    else:
        pos["position"] = float(base)
    return pos.reset_index(drop=True)


def simulate(positions_df, returns_df):
    """Hold each ticker's position over the next day, equal-weight across the universe.

    Positions are shifted forward one day per ticker so a decision at the close of day t earns
    day t+1's return; the final day's position never gets paid.
    """
    df = positions_df.merge(returns_df, on=["date", "ticker"], how="inner")
    df = df.sort_values(["ticker", "date"])
    df["held"] = df.groupby("ticker")["position"].shift(1)
    df = df.dropna(subset=["held"])

    # unallocated capital sits in cash at 0%, so we divide by the full universe size
    n = df["ticker"].nunique()
    contrib = df["held"] * df["ret"]
    port = contrib.groupby(df["date"]).sum() / n

    equity = pd.DataFrame({"date": port.index, "port_ret": port.to_numpy()})
    equity["equity"] = (1.0 + equity["port_ret"]).cumprod()
    equity["drawdown"] = equity["equity"] / equity["equity"].cummax() - 1.0

    r = equity["port_ret"]
    total_return = float(equity["equity"].iloc[-1] - 1.0) if len(equity) else 0.0
    ann_return = float((1.0 + r.mean()) ** TRADING_DAYS - 1.0)
    ann_vol = float(r.std() * np.sqrt(TRADING_DAYS))
    sharpe = float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else 0.0
    metrics = {
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": float(equity["drawdown"].min()),
        "avg_exposure": float(df["held"].mean()),
        "n_days": int(len(equity)),
    }
    return equity.reset_index(drop=True), metrics
