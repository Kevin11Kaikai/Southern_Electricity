"""Compact day-level price heads for the holdout RMSE bakeoff.

These models map a 96-step day of feature channels to 96 absolute prices.
They are a new scoring module and do not reuse nested or action-head
Transformer code from other experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


STEPS_PER_DAY = 96


class CompactDayLSTM(nn.Module):
    """One-layer LSTM over a 96-step day; features are input channels."""

    def __init__(self, n_features: int, hidden: int = 32) -> None:
        super().__init__()
        self.proj = nn.Linear(n_features, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.proj(x))
        output, _ = self.lstm(hidden)
        return self.head(output).squeeze(-1)


class MovingAvg(nn.Module):
    """Same-length moving average used by DLinear trend decomposition."""

    def __init__(self, kernel_size: int = 25) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("moving-average kernel_size must be odd")
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        padded = torch.cat([front, x, end], dim=1)
        return self.avg(padded.permute(0, 2, 1)).permute(0, 2, 1)


class CompactDLinear(nn.Module):
    """Trend/seasonal linear map over a 96-step day (Zeng et al. style, compact).

    Input is ``(batch, 96, channels)``. A shared time-linear layer is applied to
    the decomposed trend and remainder, then channels are reduced to one price.
    This is a small control model, not a copy of an external DLinear codebase.
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int = STEPS_PER_DAY,
        pred_len: int = STEPS_PER_DAY,
        kernel_size: int = 25,
    ) -> None:
        super().__init__()
        self.decomp = MovingAvg(kernel_size=kernel_size)
        self.seasonal = nn.Linear(seq_len, pred_len)
        self.trend = nn.Linear(seq_len, pred_len)
        avg_weight = torch.full((pred_len, seq_len), 1.0 / float(seq_len))
        with torch.no_grad():
            self.seasonal.weight.copy_(avg_weight)
            self.trend.weight.copy_(avg_weight)
            self.seasonal.bias.zero_()
            self.trend.bias.zero_()
        self.head = nn.Identity() if n_features == 1 else nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.decomp(x)
        seasonal = x - trend
        seasonal_out = self.seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
        trend_out = self.trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
        combined = seasonal_out + trend_out
        if isinstance(self.head, nn.Identity):
            return combined.squeeze(-1)
        return self.head(combined).squeeze(-1)


class CompactDayTransformer(nn.Module):
    """Two-layer Transformer encoder over a 96-step day; linear price head."""

    def __init__(
        self,
        n_features: int,
        d_model: int = 32,
        nhead: int = 4,
        nlayers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, STEPS_PER_DAY, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.proj(x) + self.pos
        hidden = self.encoder(hidden)
        return self.head(hidden).squeeze(-1)


@dataclass(frozen=True)
class SequenceTrainResult:
    model: nn.Module
    best_val_rmse: float
    epochs_run: int
    best_epoch: int
    fill_values: np.ndarray
    mean: np.ndarray
    scale: np.ndarray


def _standardize(
    values: np.ndarray,
    fill_values: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    filled = np.where(np.isfinite(values), values, fill_values)
    return ((filled - mean) / scale).astype(np.float32, copy=False)


def fit_preprocess_stats(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median fill and z-score stats from training days only, shape (F,)."""

    flat = train_x.reshape(-1, train_x.shape[-1])
    fill_values = np.nanmedian(flat, axis=0)
    fill_values = np.where(np.isfinite(fill_values), fill_values, 0.0).astype(np.float32)
    filled = np.where(np.isfinite(flat), flat, fill_values)
    mean = filled.mean(axis=0).astype(np.float32)
    scale = filled.std(axis=0).astype(np.float32)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return fill_values, mean, scale


def apply_preprocess(
    values: np.ndarray,
    fill_values: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return _standardize(values, fill_values, mean, scale)


def _rmse(yhat: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((yhat - y) ** 2)))


def train_sequence_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int = 42,
    max_epochs: int = 40,
    patience: int = 6,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cpu",
) -> SequenceTrainResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(False)

    fill_values, mean, scale = fit_preprocess_stats(x_train)
    x_train_z = apply_preprocess(x_train, fill_values, mean, scale)
    x_val_z = apply_preprocess(x_val, fill_values, mean, scale)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_state: dict[str, torch.Tensor] | None = None
    best_rmse = float("inf")
    best_epoch = 0
    wait = 0
    epochs_run = 0
    n_train = int(x_train_z.shape[0])
    x_val_t = torch.from_numpy(x_val_z).to(device)

    for epoch in range(max_epochs):
        model.train()
        permutation = np.random.permutation(n_train)
        for start in range(0, n_train, batch_size):
            index = permutation[start : start + batch_size]
            batch_x = torch.from_numpy(x_train_z[index]).to(device)
            batch_y = torch.from_numpy(y_train[index]).to(device)
            prediction = model(batch_x)
            loss = F.mse_loss(prediction, batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_t).detach().cpu().numpy()
        val_rmse = _rmse(val_pred, y_val)
        epochs_run = epoch + 1
        if val_rmse < best_rmse - 1e-6:
            best_rmse = val_rmse
            best_epoch = epochs_run
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is None:
        raise RuntimeError("sequence model produced no training state")
    model.load_state_dict(best_state)
    model.eval()
    return SequenceTrainResult(
        model=model,
        best_val_rmse=float(best_rmse),
        epochs_run=int(epochs_run),
        best_epoch=int(best_epoch),
        fill_values=fill_values,
        mean=mean,
        scale=scale,
    )


def train_sequence_fixed_epochs(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    num_epochs: int,
    seed: int = 42,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cpu",
) -> SequenceTrainResult:
    """Retrain on all development days for a frozen epoch count (no holdout val)."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    fill_values, mean, scale = fit_preprocess_stats(x_train)
    x_train_z = apply_preprocess(x_train, fill_values, mean, scale)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_train = int(x_train_z.shape[0])
    epochs = max(1, int(num_epochs))
    for _epoch in range(epochs):
        model.train()
        permutation = np.random.permutation(n_train)
        for start in range(0, n_train, batch_size):
            index = permutation[start : start + batch_size]
            batch_x = torch.from_numpy(x_train_z[index]).to(device)
            batch_y = torch.from_numpy(y_train[index]).to(device)
            prediction = model(batch_x)
            loss = F.mse_loss(prediction, batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    return SequenceTrainResult(
        model=model,
        best_val_rmse=float("nan"),
        epochs_run=epochs,
        best_epoch=epochs,
        fill_values=fill_values,
        mean=mean,
        scale=scale,
    )


def predict_sequence(
    result: SequenceTrainResult,
    x_test: np.ndarray,
    *,
    device: str = "cpu",
) -> np.ndarray:
    x_test_z = apply_preprocess(x_test, result.fill_values, result.mean, result.scale)
    model = result.model.to(device)
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x_test_z).to(device)).detach().cpu().numpy()
