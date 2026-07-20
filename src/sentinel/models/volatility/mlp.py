"""Feed-forward classifier for the volatility regime, trained with focal loss."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from sentinel.models.volatility.losses import focal_loss


class MLP(nn.Module):
    def __init__(self, n_features, hidden=(64, 32), dropout=0.3):
        super().__init__()
        layers, prev = [], n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _device(device):
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def train_mlp(train, valid=None, epochs=40, lr=1e-3, batch_size=256,
              alpha=0.25, gamma=2.0, seed=42, device=None):
    """Train the MLP. With a validation set, keep the best PR-AUC epoch; without one
    (e.g. inside a walk-forward fold), just train the full number of epochs."""
    from sklearn.metrics import average_precision_score
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = _device(device)

    Xtr, ytr = train
    model = MLP(Xtr.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr_t = torch.tensor(Xtr, device=dev)
    ytr_t = torch.tensor(ytr, device=dev)

    track = valid is not None
    if track:
        Xva, yva = valid
        Xva_t = torch.tensor(Xva, device=dev)
    n, best_ap, best_state = len(Xtr_t), -1.0, None

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            focal_loss(model(Xtr_t[idx]), ytr_t[idx], alpha, gamma).backward()
            opt.step()
        if track:
            model.eval()
            with torch.no_grad():
                ap = average_precision_score(yva, torch.sigmoid(model(Xva_t)).cpu().numpy())
            if ap > best_ap:
                best_ap = ap
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if track and best_state is not None:
        model.load_state_dict(best_state)
    return model


def mlp_proba(model, X, device=None):
    dev = _device(device)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, device=dev))).cpu().numpy()
