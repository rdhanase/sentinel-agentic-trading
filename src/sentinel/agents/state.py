"""Shared state passed between the trading-agent nodes.

Kept as a plain TypedDict (total=False) so nodes can return partial updates and
LangGraph merges them. Everything is keyed by ticker to stay decoupled from the
order of the universe.
"""
from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class TradingState(TypedDict, total=False):
    date: Any                              # trading day being decided (pd.Timestamp)
    universe: List[str]                    # tickers we actually have data for that day
    gate_on: bool                          # whether the volatility circuit-breaker is live
    t_scale: float                         # proba >= this -> half size
    t_halt: float                          # proba >= this -> flat

    market: Dict[str, Dict[str, float]]    # ticker -> features + "proba"
    sentiment: Dict[str, float]            # ticker -> directional score in [-1, 1]
    volatility: Dict[str, Dict[str, Any]]  # ticker -> {"proba", "risk_mode"}
    risk_mode: str                         # market-wide mode driving the conditional edge

    positions: Dict[str, float]            # ticker -> fraction of its slice, in [0, 1]
    executed: bool                         # True once the (paper) execution node ran

    rationale: str                         # plain-language explanation of the day's call
    trace: List[str]                       # node names in the order they fired
    log: List[dict]                        # decision records (paper trail)
