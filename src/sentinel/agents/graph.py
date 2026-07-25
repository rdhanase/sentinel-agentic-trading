r"""Assemble the nodes into the gated LangGraph decision flow.

    market -> sentiment -> risk --(conditional)--> portfolio -> execution -> rationale -> END
                                \--------------------------> halt --------> rationale -> END

The volatility circuit-breaker is a real conditional edge: when the market-wide
risk mode is HALT (and the gate is on) we route straight to ``halt``, which goes
flat and logs without ever touching the portfolio or execution nodes.
"""
from __future__ import annotations

import pandas as pd
from langgraph.graph import StateGraph, START, END

from sentinel.agents.state import TradingState
from sentinel.agents import nodes


def _route(state: TradingState) -> str:
    """Circuit-breaker branch: halt only when the gate is on and the market is HALT."""
    if state.get("gate_on", True) and state.get("risk_mode") == "HALT":
        return "halt"
    return "portfolio"


def build_graph(gate_on=True):
    g = StateGraph(TradingState)
    g.add_node("market", nodes.market_node)
    g.add_node("sentiment", nodes.sentiment_node)
    g.add_node("risk", nodes.risk_node)
    g.add_node("portfolio", nodes.portfolio_node)
    g.add_node("execution", nodes.execution_node)
    g.add_node("halt", nodes.halt_node)
    g.add_node("rationale", nodes.rationale_node)

    g.add_edge(START, "market")
    g.add_edge("market", "sentiment")
    g.add_edge("sentiment", "risk")
    g.add_conditional_edges("risk", _route, {"portfolio": "portfolio", "halt": "halt"})
    g.add_edge("portfolio", "execution")
    g.add_edge("execution", "rationale")
    g.add_edge("halt", "rationale")
    g.add_edge("rationale", END)

    return g.compile()


def _init_state(date, universe, gate_on):
    state = {"date": pd.Timestamp(date), "gate_on": gate_on, "log": [], "trace": []}
    if universe is not None:
        state["universe"] = list(universe)
    return state


def run_day(date, universe=None, gate_on=True):
    """Single-day traced run -- returns the full final state (rationale + log)."""
    graph = build_graph(gate_on=gate_on)
    return graph.invoke(_init_state(date, universe, gate_on))


def run_backtest(gate_on=True, universe=None):
    """Walk every out-of-sample date and emit the positions the backtest consumes:
    columns [date, ticker, position], one row per held name per day."""
    graph = build_graph(gate_on=gate_on)
    rows = []
    for date in nodes.trading_dates():
        state = graph.invoke(_init_state(date, universe, gate_on))
        for tk, w in state.get("positions", {}).items():
            rows.append({"date": date, "ticker": tk, "position": float(w)})
    return pd.DataFrame(rows, columns=["date", "ticker", "position"])
