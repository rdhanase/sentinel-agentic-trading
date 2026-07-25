import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.sim.backtest import build_positions, gate_multiplier, risk_mode, simulate


def test_gate_multiplier_thresholds():
    assert gate_multiplier(0.39) == 1.0 and risk_mode(0.39) == "NORMAL"
    assert gate_multiplier(0.40) == 0.5 and risk_mode(0.40) == "SCALED"
    assert gate_multiplier(0.69) == 0.5
    assert gate_multiplier(0.70) == 0.0 and risk_mode(0.70) == "HALT"


def test_gate_off_keeps_full_book():
    oos = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"] * 2),
                        "ticker": ["A", "B"], "proba": [0.9, 0.1]})
    assert (build_positions(oos, gate_on=False)["position"] == 1.0).all()
    gated = build_positions(oos, gate_on=True).set_index("ticker")["position"]
    assert gated["A"] == 0.0 and gated["B"] == 1.0


def test_simulate_known_equity():
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    pos = pd.DataFrame({"date": dates, "ticker": "A", "position": 1.0})
    rets = pd.DataFrame({"date": dates, "ticker": "A", "ret": [0.10, 0.10, -0.05]})
    equity, m = simulate(pos, rets)
    # single ticker, full position: portfolio return is just that ticker's return each held day
    np.testing.assert_allclose(equity["port_ret"].to_numpy(), [0.10, -0.05])
    np.testing.assert_allclose(equity["equity"].to_numpy(), [1.10, 1.10 * 0.95])
    assert m["n_days"] == 2


def test_one_day_alignment():
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    # position only on day 1; it should earn day 2's return and nothing else
    pos = pd.DataFrame({"date": dates, "ticker": "A", "position": [1.0, 0.0, 0.0]})
    rets = pd.DataFrame({"date": dates, "ticker": "A", "ret": [0.99, 0.20, 0.99]})
    equity, _ = simulate(pos, rets)
    # day 1's return (0.99) is never earned; day 1's position earns day 2 (0.20); last day pays nothing
    np.testing.assert_allclose(equity["port_ret"].to_numpy(), [0.20, 0.0])
