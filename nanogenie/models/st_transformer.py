import torch

from einops import rearrange
from torch import nn


class STTransformer(nn.Module):
    """
    ST-Transformer architecture.
    Consists on stacking num_blocks blocks of STTransformerBlock.
    """

    def __init__(
        self,
        num_layers: int = 12,
        d_model: int = 512,
        num_heads: int = 8,
        ffn_dim: int = 2048,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                STTransformerBlock(
                    d_model=d_model, num_heads=num_heads, ffn_dim=ffn_dim
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, x: torch.Tensor, causal_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass through ST-Transformer.

        Args:
            x: Input tensor of shape [B, T, H, W, D]
            causal_mask: Optional causal mask. If None, creates one automatically.

        Returns:
            Output tensor of shape [B, T, H, W, D]
        """
        # Create causal mask if not provided
        if causal_mask is None:
            T = x.shape[1]
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device), diagonal=1
            ).bool()

        x_out = x
        for block in self.blocks:
            x_out = block(x=x_out, causal_mask=causal_mask)

        return x_out


class STTransformerBlock(nn.Module):
    """
    Each ST-Transformer block has:
    1. Spatial attention
    2. Temporal attention
    3. FFW after spatial + temporal layers.
    """

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int):
        """
        Initialize ST-Transformer block.

        Args:
            d_model: Embedding dimension
            num_heads: Number of attention heads
            ffn_dim: Hidden dimension of feed-forward network
        """
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
        Forward pass through ST-Transformer block.

        Args:
            x: Input tensor of shape [B, T, H, W, D]
            causal_mask: Optional causal mask for temporal attention of shape [T, T]

        Returns:
            Output tensor of shape [B, T, H, W, D]
        """
        ### 1. Spatial Attention ##
        x_spatial = self.spatial_attention(x)  # [B, T, H, W, D]

        ### 2. Temporal attention ###
        x_temporal = self.temporal_attention(x_spatial, causal_mask)  # [B, T, H, W, D]

        ### 3. Feed-forward network ###
        ffn_out = self.feed_forward(x_temporal)

        return ffn_out

    def spatial_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial self-attention across spatial dimensions (H, W) for each frame.

        Args:
            x: Input tensor of shape [B, T, H, W, D]

        Returns:
            Output tensor of shape [B, T, H, W, D]
        """
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

        return x_spatial

    def temporal_attention(
        self, x: torch.Tensor, causal_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply causal temporal self-attention across time dimension (T) for each spatial location.

        Args:
            x: Input tensor of shape [B, T, H, W, D]
            causal_mask: Causal mask of shape [T, T] to prevent attending to future frames

        Returns:
            Output tensor of shape [B, T, H, W, D]
        """
        B, T, H, W, D = x.shape
        # Flatten for multi-head attention
        x_temporal = rearrange(
            x, "b t h w d -> (b h w) t d"
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

        return x_temporal

    def feed_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feed-forward network with residual connection and layer normalization.

        Args:
            x: Input tensor of shape [B, T, H, W, D]

        Returns:
            Output tensor of shape [B, T, H, W, D]
        """
        ffn_out = self.ffn(x)
        ffn_out = self.ffn_norm(x + ffn_out)  # [B, T, H, W, D]
        return ffn_out
