from __future__ import annotations

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """LSTM baseline for multivariate PPG feature sequences.

    Expected input shape is [batch, time, channels], matching the ResNet-1D
    training pipeline in this repository.
    """

    def __init__(
        self,
        input_dim: int = 17,
        output_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        lstm_dropout = float(dropout) if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        direction_factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(hidden_dim * direction_factor, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, channels], got shape={tuple(x.shape)}")
        _, (hidden, _) = self.lstm(x)
        if self.lstm.bidirectional:
            last = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            last = hidden[-1]
        return self.classifier(self.dropout(last))
