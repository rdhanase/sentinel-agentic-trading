"""Fine-tune FinBERT for three-way financial sentiment."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import LABELS, ID2LABEL, LABEL2ID, load_phrasebank

MODEL_NAME = "ProsusAI/finbert"


def _metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro")}


def finetune(output_dir="models_store/finbert-sentiment", agreement="sentences_66agree",
             epochs=4, lr=2e-5, batch_size=16, max_length=128, seed=42):
    """Fine-tune FinBERT on the Financial PhraseBank and return the trainer, the tokenized
    splits, and the held-out test metrics."""
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer, DataCollatorWithPadding)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    # the base head is FinBERT's own 3-class one; fine-tuning realigns it to our label order
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)

    data = load_phrasebank(agreement=agreement, seed=seed)
    enc = data.map(lambda b: tok(b["text"], truncation=True, max_length=max_length), batched=True)

    args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=epochs, learning_rate=lr,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size * 2,
        weight_decay=0.01, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        logging_steps=50, seed=seed, report_to="none")

    trainer = Trainer(model=model, args=args, train_dataset=enc["train"],
                      eval_dataset=enc["validation"], tokenizer=tok,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=_metrics)
    trainer.train()
    metrics = trainer.evaluate(enc["test"])

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)
    return trainer, enc, metrics
