import torch

from einops import rearrange
from torch import nn


class STTransformerBlock(nn.Module):
    """
    Each ST-Transformer block has:
    1. Spatial attention
    2. Temporal attention
    3. FFW after spatial + temporal layers.
    """

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int):
        super().__init__()
        # Spatial attention
        # [B, T, H, W, d_model] -> [B*T, H*W, d_model]
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True
        )
        self.spatial_norm = nn.LayerNorm(d_model)

        # Temporal attention
        # [B, T, H, W, d_model] -> [B*H*W, T, d_model]
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True
        )
        self.temporal_norm = nn.LayerNorm(d_model)

        # Feed-forward. Will use Swish as there was no specific mention in the paper.
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.SiLU(), nn.Linear(ffn_dim, d_model)
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, causal_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """
        x: [B, T, H, W, d_model]

        """
        ### 1. Spatial Attention ##

        # Get input dimensions
        B, T, H, W, D = x.shape

        # Rearrange for multi-head attention frame processing in parallel
        x_spatial = rearrange(
            x, "b t h w d -> (b t) (h w) d"
        )  # [B, T, H, W, D] -> [B*T, H*W, D]

        # Spatial attention
        attn_out, _ = self.spatial_attn(x_spatial, x_spatial, x_spatial)

        # Residual + layer norm
        x_spatial = self.spatial_norm(x_spatial + attn_out)

        # Reshape back
        x_spatial = rearrange(
            x_spatial, "(b t) (h w) d -> b t h w d", b=B, t=T, h=H, w=W
        )  # [B*T, H*W, D] -> [B, T, H, W, D]

        ### 2. Temporal attention ###

        # Flatten for multi-head attention
        x_temporal = rearrange(
            x_spatial, "b t h w d -> (b h w) t d"
        )  # [B, T, H, W, D] -> [B*H*W, T, D]

        # Pass causal mask to temporal attention
        attn_out, _ = self.temporal_attn(
            x_temporal, x_temporal, x_temporal, attn_mask=causal_mask
        )

        # Residual + layer norm
        x_temporal = self.temporal_norm(x_temporal + attn_out)

        # Reshape back
        x_temporal = rearrange(
            x_temporal, "(b h w) t d -> b t h w d", b=B, h=H, w=W
        )  # [B*H*W, T, D] -> [B, T, H, W, D]

        ### 3. Feed-forward network ###

        ffn_out = self.ffn(x_temporal)
        ffn_out = self.ffn_norm(x_temporal + ffn_out)  # [B, T, H, W, D]

        return ffn_out
