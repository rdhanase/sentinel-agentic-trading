"""Load and prepare financial-sentiment corpora for fine-tuning FinBERT."""
from __future__ import annotations

LABELS = ["negative", "neutral", "positive"]
ID2LABEL = dict(enumerate(LABELS))
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# the twitter set encodes 0=bearish, 1=bullish, 2=neutral; map it onto our scheme
_TWITTER_MAP = {0: 0, 1: 2, 2: 1}

_STR2ID = {"negative": 0, "neutral": 1, "positive": 2}
_AGREE_FILE = {
    "sentences_50agree": "Sentences_50Agree.txt",
    "sentences_66agree": "Sentences_66Agree.txt",
    "sentences_75agree": "Sentences_75Agree.txt",
    "sentences_allagree": "Sentences_AllAgree.txt",
}


def load_phrasebank(agreement="sentences_66agree", val_size=0.1, test_size=0.1, seed=42):
    # Read the Financial PhraseBank straight from its zip and split it here. This skips the old
    # dataset loading script, which recent datasets / huggingface_hub releases no longer accept.
    import zipfile
    from huggingface_hub import hf_hub_download
    from sklearn.model_selection import train_test_split
    from datasets import Dataset, DatasetDict

    zip_path = hf_hub_download("takala/financial_phrasebank",
                               "data/FinancialPhraseBank-v1.0.zip", repo_type="dataset")
    member = f"FinancialPhraseBank-v1.0/{_AGREE_FILE[agreement]}"
    texts, labels = [], []
    with zipfile.ZipFile(zip_path) as z:
        for line in z.read(member).decode("latin-1").splitlines():
            if "@" not in line:
                continue
            sentence, tag = line.rsplit("@", 1)
            texts.append(sentence.strip())
            labels.append(_STR2ID[tag.strip()])

    idx = list(range(len(texts)))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed,
                                           stratify=labels)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_size / (1 - test_size),
                                          random_state=seed, stratify=[labels[i] for i in train_idx])
    ds = Dataset.from_dict({"text": texts, "labels": labels})
    return DatasetDict(train=ds.select(train_idx), validation=ds.select(val_idx),
                       test=ds.select(test_idx))


def load_twitter_ood():
    # informal tweets, kept as an out-of-distribution check rather than for training
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from datasets import Dataset

    frames = []
    for fn in ["sent_train.csv", "sent_valid.csv"]:
        path = hf_hub_download("zeroshot/twitter-financial-news-sentiment", fn, repo_type="dataset")
        frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)
    df["labels"] = df["label"].map(_TWITTER_MAP)
    return Dataset.from_pandas(df[["text", "labels"]], preserve_index=False)
