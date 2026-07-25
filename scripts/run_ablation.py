"""Circuit-breaker ablation: does gating the equal-weight book on the volatility model help?

    python scripts/run_ablation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel.sim.backtest import build_positions, simulate

FIG = ROOT / "reports" / "figures"


def main():
    oos = pd.read_parquet(ROOT / "data/processed/volatility_oos.parquet")
    feats = pd.read_parquet(ROOT / "data/processed/features_labeled.parquet",
                            columns=["date", "ticker", "ret_1"])
    feats["ret"] = np.expm1(feats["ret_1"])  # ret_1 is a daily log return
    returns = feats[["date", "ticker", "ret"]]

    runs = {}
    for gate_on in (False, True):
        pos = build_positions(oos, gate_on=gate_on)
        equity, metrics = simulate(pos, returns)
        runs["gate_on" if gate_on else "gate_off"] = (equity, metrics)

    print(f"\nAblation {oos['date'].min().date()} .. {oos['date'].max().date()} | "
          f"{oos['ticker'].nunique()} tickers")
    header = f"{'':<10}{'total_ret':>12}{'sharpe':>10}{'max_dd':>10}{'avg_exp':>10}"
    print(header)
    print("-" * len(header))
    for name, (_, m) in runs.items():
        print(f"{name:<10}{m['total_return']:>12.2%}{m['sharpe']:>10.2f}"
              f"{m['max_drawdown']:>10.2%}{m['avg_exposure']:>10.2f}")

    FIG.mkdir(parents=True, exist_ok=True)
    _plot(runs, "equity", "cumulative equity",
          "Circuit-breaker ablation: equity", FIG / "ablation_equity.png")
    _plot(runs, "drawdown", "drawdown",
          "Circuit-breaker ablation: drawdown", FIG / "ablation_drawdown.png")
    print(f"\nfigures -> {FIG.relative_to(ROOT)}/")


def _plot(runs, col, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, (equity, _) in runs.items():
        ax.plot(equity["date"], equity[col], label=name)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
