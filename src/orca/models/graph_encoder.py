"""Pure PyTorch GATv2 Graph Neural Network for Microservice Service Dependencies."""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATv2Layer(nn.Module):
    """Vectorized multi-head GATv2 layer operating on dense adjacency matrices."""

    def __init__(self, in_features: int, out_features: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.head_dim = out_features // heads

        self.linear_src = nn.Linear(in_features, out_features, bias=False)
        self.linear_dst = nn.Linear(in_features, out_features, bias=False)
        self.attn_vec = nn.Parameter(torch.empty(1, heads, 1, self.head_dim))
        nn.init.xavier_uniform_(self.attn_vec)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features of shape (B, N, in_features)
            adj: Adjacency matrix of shape (B, N, N)
            
        Returns:
            out: Updated node features of shape (B, N, out_features)
        """
        B, N, _ = x.shape
        res = self.residual(x)

        # Linear projections
        src = self.linear_src(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        dst = self.linear_dst(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)

        # Pairwise combination for GATv2: LeakyReLU(W_src x_i + W_dst x_j)
        pair = self.leaky_relu(src.unsqueeze(3) + dst.unsqueeze(2))  # (B, H, N, N, D)

        # Compute attention scores: (B, H, N, N)
        scores = (pair * self.attn_vec.unsqueeze(2)).sum(dim=-1)

        # Mask non-edges where adj == 0 (with large negative value)
        adj_expanded = adj.unsqueeze(1)  # (B, 1, N, N)
        mask = (adj_expanded <= 1e-6)
        scores = scores.masked_fill(mask, -1e9)

        # Softmax over target nodes
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Message aggregation: (B, H, N, N) @ (B, H, N, D) -> (B, H, N, D)
        out = torch.matmul(attn_weights, dst)
        out = out.transpose(1, 2).contiguous().view(B, N, self.out_features)

        return F.gelu(out + res)


class ServiceGraphEncoder(nn.Module):
    """Encodes microservice call graph topology and trace attributes via GATv2."""

    def __init__(
        self,
        node_feat_dim: int = 27,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        trace_stats_dim: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_proj = nn.Linear(node_feat_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            GATv2Layer(hidden_dim, hidden_dim, heads=heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.trace_proj = nn.Sequential(
            nn.Linear(trace_stats_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        
        self.graph_pool_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self, graph_nodes: torch.Tensor, graph_adj: torch.Tensor, trace_stats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            graph_nodes: (B, 10, node_feat_dim)
            graph_adj: (B, 10, 10)
            trace_stats: (B, 4)
            
        Returns:
            z_graph: (B, hidden_dim) global graph representation
            node_embeds: (B, 10, hidden_dim) per-service node embeddings
        """
        h = self.node_proj(graph_nodes)
        for layer in self.layers:
            h = layer(h, graph_adj)

        # Global graph pooling (mean across 10 service nodes)
        mean_pooled = h.mean(dim=1)
        h_stats = self.trace_proj(trace_stats)
        
        z_graph = self.graph_pool_proj(torch.cat([mean_pooled, h_stats], dim=-1))
        return z_graph, h
