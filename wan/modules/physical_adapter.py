import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicalAdapter(nn.Module):
    def __init__(
        self,
        input_dim=768,
        hidden_dim=768,
        query_dim=None,
        output_channels=16,
        num_queries=128,
        num_heads=12,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # Learnable Queries: [1, N, D]
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
        self.query_proj = None
        if query_dim is not None and query_dim != hidden_dim:
            self.query_proj = nn.Linear(query_dim, hidden_dim)

        # Cross Attention Layer
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            kdim=input_dim,
            vdim=input_dim,
            batch_first=True,
        )

        # Feed Forward / Projection
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.output_proj = nn.Linear(hidden_dim, output_channels)

    def forward(self, ref_features, target_shape, query_override=None):
        """
        ref_features: [B, Seq_Len, Dim]
        target_shape: (B, C, F, H, W)
        """
        batch_size = ref_features.shape[0]

        if query_override is not None:
            queries = query_override
            if self.query_proj is not None:
                queries = self.query_proj(queries)
        else:
            queries = self.queries.expand(batch_size, -1, -1)
        ref_features = ref_features.to(queries.dtype)

        attn_output, _ = self.cross_attn(
            query=queries,
            key=ref_features,
            value=ref_features,
        )

        x = self.ff(attn_output) + attn_output
        x = self.output_proj(x)

        # Return token sequence directly: [B, num_queries, output_channels]
        return x
