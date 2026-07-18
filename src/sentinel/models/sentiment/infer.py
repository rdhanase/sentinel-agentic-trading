"""Turn text into the sentiment signal the trading agent consumes."""
from __future__ import annotations

import numpy as np

from .data import LABELS

_POS = LABELS.index("positive")
_NEG = LABELS.index("negative")


def sentiment_score(probs):
    """Directional signal in [-1, 1]: P(positive) minus P(negative)."""
    probs = np.asarray(probs, dtype=float)
    return probs[..., _POS] - probs[..., _NEG]


class SentimentScorer:
    def __init__(self, model_dir, device=None):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict_proba(self, texts, batch_size=32, max_length=128):
        import torch
        if isinstance(texts, str):
            texts = [texts]
        chunks = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(texts[i:i + batch_size], return_tensors="pt", truncation=True,
                           padding=True, max_length=max_length).to(self.device)
            with torch.no_grad():
                chunks.append(self.model(**enc).logits.softmax(-1).cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def score(self, texts, **kw):
        return sentiment_score(self.predict_proba(texts, **kw))
