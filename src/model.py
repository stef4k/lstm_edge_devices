"""
model.py
LSTM model for CSI-based Human Activity Recognition — PyTorch implementation.

Architecture (paper §3.3):
  Input  (T, F)
    → LSTM(lstm_units, batch_first=True)
    → take last hidden state
    → Dropout
    → Linear(n_classes)
    → Softmax (applied in loss, not here)

The paper uses 100 LSTM units for 30 subcarriers.
"""

import torch
import torch.nn as nn


class CsiLSTM(nn.Module):
    """
    Single-layer LSTM classifier for CSI time-series windows.

    Parameters
    ----------
    n_features   : int   Number of input features (subcarriers).
    n_classes    : int   Number of output classes.
    lstm_units   : int   Hidden size of the LSTM (paper: 100).
    dropout      : float Dropout probability after LSTM output.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        lstm_units: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(lstm_units, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, F)
        out, _ = self.lstm(x)      # out: (batch, T, hidden)
        last   = out[:, -1, :]    # (batch, hidden)
        last   = self.dropout(last)
        logits = self.fc(last)     # (batch, n_classes)
        return logits


def build_model(
    n_features: int,
    n_classes: int,
    lstm_units: int = 64,
    dropout: float = 0.3,
) -> CsiLSTM:
    """Construct and return the LSTM model."""
    return CsiLSTM(
        n_features=n_features,
        n_classes=n_classes,
        lstm_units=lstm_units,
        dropout=dropout,
    )


def model_summary_str(model: nn.Module, input_shape: tuple) -> str:
    """Return a brief model description string."""
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines = [
        str(model),
        f"Trainable parameters: {n_params:,}",
        f"Input shape: (batch, {input_shape[0]}, {input_shape[1]})",
    ]
    return "\n".join(lines)
