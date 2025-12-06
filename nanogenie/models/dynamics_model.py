import torch

from einops import rearrange
from nanogenie.models import STTransformer
from torch import nn


class DynamicsModel(nn.Module):
    """
    Implement the Dynamics Model described in Genie 1.

    Attributes:
        z_embedding (nn.Embedding): Embedding for discrete video tokens. [B, T-1, H', W', d_model]
        a_embedding (nn.Embedding): Embedding for the latent actions.   [B, T-1, H', W', d_model]
        st_transformer (STTransformer): The backbone processing spatial and temporal dependencies.
        output_head (nn.Linear): Projects the transformer output back to the token vocabulary.

    Args:
        vocab_z (int): Size of the video tokenizer codebook (default: 1024).
        vocab_a (int): Number of discrete latent actions (default: 6 for CoinRun, 8 for Platformers).
        d_model (int): Hidden dimension of the transformer (default: 512).
        num_layers (int): Number of ST-Transformer blocks (default: 12).
        num_heads (int): Number of attention heads (default: 8).
        ffn_dim (int): Dimension of the feed-forward network (default: 2048).
    """

    def __init__(
        self,
        vocab_z: int = 1024,
        vocab_a: int = 6,
        d_model: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        ffn_dim: int = 2048,
    ):
        super().__init__()

        # Token embeddings
        self.z_embedding = nn.Embedding(vocab_z, d_model)

        # Latent action embeddings
        self.a_embedding = nn.Embedding(vocab_a, d_model)

        # Add ST-Transformer blocks
        self.st_transformer = STTransformer(
            num_heads=num_heads,
            num_layers=num_layers,
            d_model=d_model,
            ffn_dim=ffn_dim,
        )

        # Output projection, d_model -> vocab_z
        self.output_head = nn.Linear(d_model, vocab_z)

    def forward(
        self, z: torch.Tensor, a: torch.Tensor, causal_mask: torch.Tensor
    ) -> torch.Tensor:
        # 1. Embeddings
        z_emb = self.z_embedding(z)  # [B, T-1, H', W', d_model]
        a_emb = self.a_embedding(a)  # [B, T-1, d_model]

        # 2. "Expand" a as the original action tensor is 1 action per frame only
        # [B, T-1, dim_model] -> [B, T-1, 1, 1, d_model]
        a_emb = rearrange(a_emb, "b t d -> b t 1 1 d")

        # 3. Additive embedding
        x = z_emb + a_emb  # [B, T-1, H', W', d_model]

        # 4. ST-Transformer with causal mask
        x_transf = self.st_transformer(
            x=x, causal_mask=causal_mask
        )  # [B, T-1, H', W', d_model]

        # 5. Output Head
        logits = self.output_head(x_transf)  # [B, T-1, H', W', vocab_z]

        return logits
