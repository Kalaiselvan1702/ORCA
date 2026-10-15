"""Multimodal Fusion Modules for ORCA: Cross-Attention, Gated, and Concatenation."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """Multi-head cross-attention mechanism across multimodal representations."""

    def __init__(self, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self, z_log: torch.Tensor, z_metric: torch.Tensor, z_graph: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z_log, z_metric, z_graph: (batch_size, embed_dim)
            
        Returns:
            fused: (batch_size, embed_dim)
        """
        # Stack modalities as a sequence of length 3: (B, 3, D)
        modalities = torch.stack([z_log, z_metric, z_graph], dim=1)
        
        # Self/Cross-attention over the multimodal tokens
        attn_out, weights = self.cross_attn(modalities, modalities, modalities)
        attn_out = self.norm(modalities + attn_out)  # Residual connection

        # Flatten and project
        b = z_log.size(0)
        flat = attn_out.reshape(b, -1)
        fused = self.mlp(flat)
        return fused


class GatedFusion(nn.Module):
    """Gated multimodal fusion mechanism with learnable dynamic modality routing."""

    def __init__(self, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.gate_linear = nn.Linear(embed_dim * 3, 3)
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self, z_log: torch.Tensor, z_metric: torch.Tensor, z_graph: torch.Tensor
    ) -> torch.Tensor:
        stacked = torch.cat([z_log, z_metric, z_graph], dim=-1)
        gates = F.softmax(self.gate_linear(stacked), dim=-1)  # (B, 3)

        g_log = gates[:, 0:1]
        g_metric = gates[:, 1:2]
        g_graph = gates[:, 2:3]

        fused = g_log * z_log + g_metric * z_metric + g_graph * z_graph
        return self.out_proj(fused)


class ConcatFusion(nn.Module):
    """Simple linear projection fusion baseline."""

    def __init__(self, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self, z_log: torch.Tensor, z_metric: torch.Tensor, z_graph: torch.Tensor
    ) -> torch.Tensor:
        cat = torch.cat([z_log, z_metric, z_graph], dim=-1)
        return self.proj(cat)
