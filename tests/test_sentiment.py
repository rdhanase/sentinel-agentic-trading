import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.models.sentiment.data import LABELS, LABEL2ID, _TWITTER_MAP
from sentinel.models.sentiment.infer import sentiment_score


def test_label_scheme():
    assert LABELS == ["negative", "neutral", "positive"]
    assert LABEL2ID["negative"] == 0 and LABEL2ID["positive"] == 2


def test_twitter_mapping():
    # bearish -> negative, bullish -> positive, neutral -> neutral
    assert _TWITTER_MAP == {0: 0, 1: 2, 2: 1}


def test_sentiment_score_direction():
    assert sentiment_score([0.0, 0.0, 1.0]) == 1.0
    assert sentiment_score([1.0, 0.0, 0.0]) == -1.0
    assert abs(sentiment_score([0.1, 0.8, 0.1])) < 1e-9


def test_sentiment_score_batch():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
    s = sentiment_score(probs)
    assert s.shape == (2,)
    assert s[0] < 0 < s[1]
