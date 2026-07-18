"""Load and prepare financial-sentiment corpora for fine-tuning FinBERT."""
from __future__ import annotations

LABELS = ["negative", "neutral", "positive"]
ID2LABEL = dict(enumerate(LABELS))
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# the twitter set encodes 0=bearish, 1=bullish, 2=neutral; map it onto our scheme
_TWITTER_MAP = {0: 0, 1: 2, 2: 1}


def load_phrasebank(agreement="sentences_66agree", val_size=0.1, test_size=0.1, seed=42):
    # Financial PhraseBank ships as one split, already coded 0=neg, 1=neu, 2=pos
    from datasets import load_dataset, DatasetDict
    ds = load_dataset("takala/financial_phrasebank", agreement, trust_remote_code=True)["train"]
    ds = ds.rename_column("sentence", "text").rename_column("label", "labels")
    outer = ds.train_test_split(test_size=test_size, seed=seed, stratify_by_column="labels")
    inner = outer["train"].train_test_split(test_size=val_size / (1 - test_size), seed=seed,
                                            stratify_by_column="labels")
    return DatasetDict(train=inner["train"], validation=inner["test"], test=outer["test"])


def load_twitter_ood():
    # informal tweets, kept as an out-of-distribution check rather than for training
    from datasets import load_dataset, concatenate_datasets
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
    both = concatenate_datasets([ds["train"], ds["validation"]])
    return both.map(lambda r: {"labels": _TWITTER_MAP[r["label"]]})
