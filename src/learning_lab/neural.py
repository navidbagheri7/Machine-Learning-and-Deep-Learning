"""Residual tabular networks, feature-token attention, and explicit validation selection."""

import copy
import random

import numpy as np
import torch
from torch import nn


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
        )

    def forward(self, x):
        return x + self.net(x)


class TabularResNet(nn.Module):
    def __init__(self, features, width=128, depth=3, dropout=0.15):
        super().__init__()
        self.input = nn.Linear(features, width)
        self.blocks = nn.Sequential(*[ResidualBlock(width, dropout) for _ in range(depth)])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, x):
        return self.output(self.blocks(self.input(x))).squeeze(-1)


class FeatureTransformer(nn.Module):
    """Numerical FT-style tokens x_j*w_j+b_j followed by a CLS token readout.

    This compact numerical-feature variant is not the paper's exact benchmark implementation.
    """

    def __init__(self, features, width=48, depth=2, heads=4, dropout=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(features, width) * 0.02)
        self.bias = nn.Parameter(torch.zeros(features, width))
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(
            width, heads, width * 2, dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, x):
        tokens = x[:, :, None] * self.weight + self.bias
        tokens = torch.cat([self.cls.expand(len(x), -1, -1), tokens], dim=1)
        return self.output(self.encoder(tokens)[:, 0]).squeeze(-1)


def train_tabular(
    x, y, xv, yv, kind="resnet", task="classification", seed=0, epochs=60, device="cpu"
):
    seed_everything(seed)
    model = (TabularResNet(x.shape[1]) if kind == "resnet" else FeatureTransformer(x.shape[1])).to(
        device
    )
    xs, ys = (
        torch.as_tensor(x, dtype=torch.float32, device=device),
        torch.as_tensor(y, dtype=torch.float32, device=device),
    )
    vx, vy = (
        torch.as_tensor(xv, dtype=torch.float32, device=device),
        torch.as_tensor(yv, dtype=torch.float32, device=device),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()
    history, best, state, stale = [], np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(xs), device=device)
        losses = []
        for ix in order.split(512):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xs[ix]), ys[ix])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val = float(criterion(model(vx), vy))
        history.append({"epoch": epoch + 1, "train_loss": np.mean(losses), "validation_loss": val})
        if val < best:
            best, state, stale = val, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= 12:
            break
    model.load_state_dict(state)
    model.eval()
    return model, history


def neural_predict(model, x, device, classification=False):
    with torch.no_grad():
        values = torch.cat(
            [model(b) for b in torch.as_tensor(x, dtype=torch.float32, device=device).split(1024)]
        )
        if classification:
            values = values.sigmoid()
    return values.cpu().numpy()
