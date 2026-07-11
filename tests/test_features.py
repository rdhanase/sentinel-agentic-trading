"""Unit tests for the technical features."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.features.technical import log_return, rsi, macd, atr, bollinger_width


def _synthetic_close(n=300, seed=0):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.01, n)
    return pd.Series(100 * np.exp(np.cumsum(steps)))


def test_log_return_matches_definition():
    close = _synthetic_close()
    r = log_return(close, 1)
    expected = np.log(close.iloc[5] / close.iloc[4])
    assert np.isclose(r.iloc[5], expected)


def test_rsi_bounded_unit_interval():
    close = _synthetic_close()
    r = rsi(close, 14).dropna()
    assert (r >= 0).all() and (r <= 1).all()


def test_macd_hist_is_line_minus_signal():
    close = _synthetic_close()
    m = macd(close)
    assert np.allclose((m["macd_line"] - m["macd_signal"]).values, m["macd_hist"].values)


def test_atr_positive_fraction():
    close = _synthetic_close()
    high = close * 1.01
    low = close * 0.99
    a = atr(high, low, close, 14).dropna()
    assert (a > 0).all() and (a < 1).all()


def test_bollinger_width_nonnegative():
    close = _synthetic_close()
    bw = bollinger_width(close, 20, 2).dropna()
    assert (bw >= 0).all()
