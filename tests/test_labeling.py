"""Unit tests for volatility labeling and leakage-safe splits."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.features.labeling import (
    forward_realized_vol, trailing_percentile, make_labels, time_split,
)


def test_forward_vol_is_forward_looking():
    # returns are zero except a spike at index 50; forward vol at t<50 should see it, at t>50 should not
    ret = pd.Series(np.zeros(100))
    ret.iloc[50] = 0.1
    fwd = forward_realized_vol(ret, horizon=5)
    assert fwd.iloc[45] > 0        # window 46..50 includes the spike
    assert fwd.iloc[60] == 0       # window 61..65 is calm
    # last `horizon` values are undefined (no future) -> NaN
    assert fwd.iloc[-1] != fwd.iloc[-1]


def test_trailing_percentile_uses_past_only():
    s = pd.Series(np.arange(1, 301, dtype=float))
    thr = trailing_percentile(s, lookback=100, q=0.8)
    # threshold at t must be computable from values strictly before t (shifted by 1)
    manual = s.iloc[100:200].quantile(0.8)
    assert np.isclose(thr.iloc[200], manual, rtol=1e-6)


def test_label_rate_tracks_quantile():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"date": pd.date_range("2015-01-01", periods=2000, freq="B"),
                       "ret_1": rng.normal(0, 0.01, 2000)})
    out = make_labels(df, horizon=5, lookback=252, quantile=0.8)
    rate = out["label_highvol"].dropna().astype(int).mean()
    # roughly the tail mass above the 80th percentile -> about 20%
    assert 0.10 < rate < 0.30


def test_time_split_embargo_gap():
    dates = pd.Series(pd.date_range("2015-01-01", periods=2000, freq="B"))
    tr, va, te = time_split(dates, "2018-12-31", "2019-12-31", embargo_days=5)
    # no overlap and embargo leaves a gap right after each boundary
    assert not (tr & va).any() and not (va & te).any()
    d = pd.to_datetime(dates)
    gap = d[(d > pd.Timestamp("2018-12-31")) & (d <= pd.Timestamp("2019-01-05"))]
    assert not tr[gap.index].any() and not va[gap.index].any()
