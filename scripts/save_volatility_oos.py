"""Save the walk-forward out-of-sample volatility predictions for the backtest/agent to consume."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

import torch
torch.set_num_threads(1)
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel.models.volatility.walk_forward import walk_forward, mlp_fit_predict


def main():
    df = pd.read_parquet(ROOT / "data/processed/features_labeled.parquet")
    oos = walk_forward(df, mlp_fit_predict, start="2016-01-01")
    out = ROOT / "data/processed/volatility_oos.parquet"
    oos.to_parquet(out, index=False)
    print(f"saved {len(oos):,} rows -> {out.relative_to(ROOT)}")
    print(f"range {oos['date'].min().date()} .. {oos['date'].max().date()} | "
          f"tickers {oos['ticker'].nunique()} | high-vol rate {oos['label_highvol'].mean():.3f}")


if __name__ == "__main__":
    main()
