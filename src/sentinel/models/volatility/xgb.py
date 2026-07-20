"""Gradient-boosted-tree baseline for the volatility regime."""
from __future__ import annotations


def train_xgb(train, seed=42, n_estimators=400, max_depth=4, lr=0.05):
    from xgboost import XGBClassifier
    Xtr, ytr = train
    pos = float(ytr.sum())
    spw = (len(ytr) - pos) / max(pos, 1.0)  # balance the rare positive class
    clf = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr,
                        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                        eval_metric="aucpr", random_state=seed, n_jobs=1)
    clf.fit(Xtr, ytr)
    return clf


def xgb_proba(clf, X):
    return clf.predict_proba(X)[:, 1]
