"""The individual decision nodes.

Each is a plain function ``state -> partial state`` so they can be unit-tested on
their own and reused outside LangGraph. The decision path (market -> risk ->
portfolio) is fully deterministic and never imports torch; the fine-tuned FinBERT
is only touched on an explicit opt-in path in :func:`sentiment_node`.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

from sentinel.features.technical import FEATURE_COLUMNS
from sentinel.agents.state import TradingState

# repo root -> data/processed, resolved from this file so cwd doesn't matter
_PROCESSED = Path(__file__).resolve().parents[3] / "data" / "processed"

# circuit-breaker thresholds on P(next-period high volatility)
T_SCALE = 0.4
T_HALT = 0.7
_MULT = {"NORMAL": 1.0, "SCALED": 0.5, "HALT": 0.0}


def _risk_mode(proba, t_scale=T_SCALE, t_halt=T_HALT):
    if proba >= t_halt:
        return "HALT"
    if proba >= t_scale:
        return "SCALED"
    return "NORMAL"


@lru_cache(maxsize=1)
def _oos_by_date():
    oos = pd.read_parquet(_PROCESSED / "volatility_oos.parquet")
    return {d: g for d, g in oos.groupby("date")}


@lru_cache(maxsize=1)
def _features_by_date():
    feats = pd.read_parquet(_PROCESSED / "features_labeled.parquet")
    return {d: g.set_index("ticker") for d, g in feats.groupby("date")}


def default_universe():
    """All tickers that have out-of-sample predictions."""
    return sorted({t for g in _oos_by_date().values() for t in g["ticker"]})


def trading_dates():
    return sorted(_oos_by_date().keys())


def _trace(state, name):
    return state.get("trace", []) + [name]


def market_node(state: TradingState) -> TradingState:
    """Pull the day's features and volatility proba for the universe.

    Skips the disk read when ``market`` is already populated, which lets a live
    feed (or a test) inject the day's data upstream.
    """
    trace = _trace(state, "market")
    if state.get("market"):
        uni = state.get("universe") or list(state["market"].keys())
        return {"market": state["market"], "universe": uni, "trace": trace}

    date = pd.Timestamp(state["date"])
    wanted = set(state.get("universe") or default_universe())
    day = _oos_by_date().get(date)
    feats = _features_by_date().get(date)

    market = {}
    if day is not None:
        for _, row in day.iterrows():
            tk = row["ticker"]
            if tk not in wanted:
                continue
            entry = {"proba": float(row["proba"])}
            if feats is not None and tk in feats.index:
                frow = feats.loc[tk]
                entry.update({c: float(frow[c]) for c in FEATURE_COLUMNS})
                entry["close"] = float(frow["close"])
            market[tk] = entry

    return {"market": market, "universe": sorted(market), "trace": trace}


def sentiment_node(state: TradingState) -> TradingState:
    """Neutral sentiment stub.

    In production this scores a per-ticker news feed with the fine-tuned FinBERT
    (:class:`sentinel.models.sentiment.infer.SentimentScorer`) and feeds a
    directional [-1, 1] signal into sizing. A historical backtest has no aligned
    news stream, so sentiment is held flat and stays out of the decision.

    The opt-in branch below only fires when both a model directory and an actual
    per-ticker news payload are present, keeping torch off the default path.
    """
    universe = state.get("universe", [])
    scores = {tk: 0.0 for tk in universe}

    model_dir = os.environ.get("SENTINEL_SENTIMENT_MODEL")
    news = state.get("news")  # {ticker: [headlines]} if a live feed wired one up
    if model_dir and os.path.isdir(model_dir) and news:
        from sentinel.models.sentiment.infer import SentimentScorer
        scorer = SentimentScorer(model_dir)
        for tk in universe:
            if news.get(tk):
                scores[tk] = float(scorer.score(news[tk]).mean())

    return {"sentiment": scores, "trace": _trace(state, "sentiment")}


def risk_node(state: TradingState) -> TradingState:
    """The circuit-breaker. Maps each ticker's proba to a risk mode and sets a
    market-wide mode (from the average proba) that drives the conditional edge."""
    t_scale = state.get("t_scale", T_SCALE)
    t_halt = state.get("t_halt", T_HALT)
    market = state.get("market", {})

    vol = {}
    for tk, m in market.items():
        p = m["proba"]
        vol[tk] = {"proba": p, "risk_mode": _risk_mode(p, t_scale, t_halt)}

    if market:
        avg = sum(m["proba"] for m in market.values()) / len(market)
        market_mode = _risk_mode(avg, t_scale, t_halt)
    else:
        market_mode = "NORMAL"

    return {"volatility": vol, "risk_mode": market_mode, "trace": _trace(state, "risk")}


def portfolio_node(state: TradingState) -> TradingState:
    """Equal-weight long-only book. Base weight is 1.0 of each ticker's slice,
    scaled down per ticker by its risk mode when the gate is on."""
    gate_on = state.get("gate_on", True)
    vol = state.get("volatility", {})

    positions = {}
    for tk in state.get("universe", []):
        mult = _MULT[vol[tk]["risk_mode"]] if (gate_on and tk in vol) else 1.0
        positions[tk] = 1.0 * mult

    return {"positions": positions, "trace": _trace(state, "portfolio")}


def execution_node(state: TradingState) -> TradingState:
    """Paper execution: record the sized book. No real orders are ever sent."""
    positions = state.get("positions", {})
    record = {
        "node": "execution",
        "date": state.get("date"),
        "risk_mode": state.get("risk_mode"),
        "gross": round(sum(positions.values()), 4),
        "n_positions": sum(1 for w in positions.values() if w > 0),
    }
    return {
        "executed": True,
        "log": state.get("log", []) + [record],
        "trace": _trace(state, "execution"),
    }


def halt_node(state: TradingState) -> TradingState:
    """Circuit-breaker landing spot: go flat and log it, skipping execution."""
    positions = {tk: 0.0 for tk in state.get("universe", [])}
    record = {
        "node": "halt",
        "date": state.get("date"),
        "risk_mode": state.get("risk_mode"),
        "gross": 0.0,
        "n_positions": 0,
    }
    return {
        "positions": positions,
        "log": state.get("log", []) + [record],
        "trace": _trace(state, "halt"),
    }


def rationale_node(state: TradingState) -> TradingState:
    """Template-based explanation of the day's call.

    A production build would hand the same structured signals to an LLM (default
    Claude ``claude-sonnet-4-6``, provider-flexible) for the narrative; we keep it
    deterministic here so the pipeline has no API dependency.
    """
    date = pd.Timestamp(state["date"]).date() if state.get("date") else "?"
    vol = state.get("volatility", {})
    mode = state.get("risk_mode", "NORMAL")
    n = len(vol)
    avg = sum(v["proba"] for v in vol.values()) / n if n else 0.0
    scaled = sum(1 for v in vol.values() if v["risk_mode"] == "SCALED")
    halted = sum(1 for v in vol.values() if v["risk_mode"] == "HALT")
    gross = round(sum(state.get("positions", {}).values()), 2)

    if mode == "HALT" and state.get("gate_on", True):
        head = (f"{date}: circuit-breaker HALT -- avg P(high-vol)={avg:.2f} across "
                f"{n} names; book taken flat.")
    else:
        head = (f"{date}: risk mode {mode} (avg P(high-vol)={avg:.2f} across {n} names). "
                f"Held {gross:.2f} gross; {scaled} names half-sized, {halted} flat.")

    rationale = head + " Sentiment neutral (no live news feed in backtest)."
    return {"rationale": rationale, "trace": _trace(state, "rationale")}
