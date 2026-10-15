"""Transformer-based Log Encoder for ORCA."""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LogTransformerEncoder(nn.Module):
    """Deep Transformer encoder for system log tokens and service-level event statistics."""

    def __init__(
        self,
        vocab_size: int = 1000,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        log_stats_dim: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=128, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Linear projections for log tokens and service stats
        self.token_proj = nn.Linear(embed_dim, hidden_dim)
        self.stats_proj = nn.Sequential(
            nn.Linear(log_stats_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, log_tokens: torch.Tensor, log_service_stats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            log_tokens: (batch_size, seq_len)
            log_service_stats: (batch_size, 20)
            
        Returns:
            z_log: (batch_size, hidden_dim)
        """
        # (batch_size, seq_len, embed_dim)
        x = self.token_embedding(log_tokens)
        x = self.pos_encoder(x)
        
        # Transformer forward pass
        mask = (log_tokens == 0)  # True where padding
        trans_out = self.transformer(x, src_key_padding_mask=mask)
        
        # Global average pooling over non-padding tokens
        non_pad_mask = (~mask).unsqueeze(-1).float()
        sum_tokens = (trans_out * non_pad_mask).sum(dim=1)
        denom = non_pad_mask.sum(dim=1).clamp(min=1.0)
        pooled = sum_tokens / denom  # (batch_size, embed_dim)
        h_tokens = self.token_proj(pooled)
        
        # Service stats representation
        h_stats = self.stats_proj(log_service_stats)
        
        # Combine
        combined = torch.cat([h_tokens, h_stats], dim=-1)
        z_log = self.out_proj(combined)
        return z_log
