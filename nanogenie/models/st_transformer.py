from torch import nn

class STTransformer(nn.Module):
    """
    Each ST-Transformer block has:
    1. Spatial attention
    2. Temporal attention
    3. FFW after spatial + temporal layers.
    """
    def __init__(self, d_model: int, num_heads: int, ff_hidden: int):
        # Spatial attention
        # [B, T, H, W, d_model] -> [B*T, H*W, d_model]
        self.spatial_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.spatial_norm = nn.LayerNorm(d_model)

        # Temporal attention
        # [B, T, H, W, d_model] -> [B*H*W, T, d_model]
        self.temporal_attn = nn.MultiheadAttention(d_model=d_model, num_heads=num_heads, batch_first=True)
        self.temporal_norm = nn.LayerNorm(d_model)

        # Feed-forward. Will use Swish as there was no specific mention in the paper.
        self.ff = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=ff_hidden),
            nn.SiLU(),
            nn.Linear(in_features=ff_hidden, out_features=d_model)
        )
        self.ff_norm = nn.LayerNorm(d_model)