"""Temporal Metric Encoder (TCN + GRU) for multi-channel time-series telemetry."""
import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    """Dilated 1D causal/temporal convolutional block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.downsample(x)
        out = self.dropout1(self.relu1(self.conv1(x)))
        out = self.dropout2(self.relu2(self.conv2(out)))
        # Adjust spatial dimension if slight mismatch due to odd padding
        if out.size(-1) != res.size(-1):
            out = out[:, :, :res.size(-1)]
        return nn.functional.gelu(out + res)


class MetricTemporalEncoder(nn.Module):
    """Encodes continuous multi-service metric time-series via TCN and bidirectional GRU."""

    def __init__(
        self,
        in_features: int = 70,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Input shape: (B, T, C) -> transpose to (B, C, T) for Conv1D
        self.tcn = nn.Sequential(
            TemporalBlock(in_features, hidden_dim // 2, kernel_size=3, dilation=1, dropout=dropout),
            TemporalBlock(hidden_dim // 2, hidden_dim, kernel_size=3, dilation=2, dropout=dropout),
        )
        
        # Temporal recurrence over the TCN features
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, metric_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            metric_seq: (batch_size, seq_len, in_features)
            
        Returns:
            z_metric: (batch_size, hidden_dim)
        """
        # Transpose for Conv1d: (B, C, T)
        x = metric_seq.transpose(1, 2)
        conv_out = self.tcn(x)
        # Transpose back for GRU: (B, T, hidden_dim)
        conv_out = conv_out.transpose(1, 2)
        
        gru_out, h_n = self.gru(conv_out)
        
        # Global temporal pooling (mean + last step)
        mean_pooled = gru_out.mean(dim=1)
        last_step = gru_out[:, -1, :]
        z = self.out_proj((mean_pooled + last_step) / 2.0)
        return z
