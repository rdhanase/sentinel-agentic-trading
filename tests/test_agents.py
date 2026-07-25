"""Tests for the LangGraph trading-agent pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.agents import nodes
from sentinel.agents.graph import build_graph, run_day, _route


def _market(probas):
    return {tk: {"proba": p} for tk, p in probas.items()}


def _state(probas, gate_on=True):
    # market pre-populated so the graph skips the parquet read and stays deterministic
    return {"market": _market(probas), "universe": sorted(probas),
            "gate_on": gate_on, "log": [], "trace": []}


def test_risk_node_maps_proba_to_mode():
    out = nodes.risk_node({"market": _market({"A": 0.3, "B": 0.5, "C": 0.8})})
    modes = {tk: v["risk_mode"] for tk, v in out["volatility"].items()}
    assert modes == {"A": "NORMAL", "B": "SCALED", "C": "HALT"}


def test_gate_off_gives_full_positions():
    graph = build_graph(gate_on=False)
    out = graph.invoke(_state({"A": 0.3, "B": 0.5, "C": 0.8}, gate_on=False))
    assert out["positions"] == {"A": 1.0, "B": 1.0, "C": 1.0}


def test_gate_on_scales_and_halts_per_ticker():
    graph = build_graph(gate_on=True)
    out = graph.invoke(_state({"A": 0.3, "B": 0.5, "C": 0.8}))
    assert out["positions"] == {"A": 1.0, "B": 0.5, "C": 0.0}


def test_conditional_edge_routes():
    assert _route({"gate_on": True, "risk_mode": "HALT"}) == "halt"
    assert _route({"gate_on": True, "risk_mode": "SCALED"}) == "portfolio"
    assert _route({"gate_on": False, "risk_mode": "HALT"}) == "portfolio"


def test_halt_skips_execution():
    graph = build_graph(gate_on=True)
    out = graph.invoke(_state({"A": 0.9, "B": 0.85, "C": 0.8}))
    assert out["risk_mode"] == "HALT"
    assert "halt" in out["trace"] and "execution" not in out["trace"]
    assert out.get("executed") is not True
    assert set(out["positions"].values()) == {0.0}
    assert out["log"]  # halt still logs a flat decision


def test_run_day_produces_rationale_and_log():
    date = nodes.trading_dates()[100]
    out = run_day(date)
    assert out["rationale"]
    assert out["log"]
    assert "execution" in out["trace"] or "halt" in out["trace"]
