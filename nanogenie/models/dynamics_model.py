import torch

from einops import rearrange
from nanogenie.models import STTransformer
from torch import nn


class DynamicsModel(nn.Module):
    """
    Implement the Dynamics Model described in Genie 1.
    It is a decoder-only MaskGIT.

    Input:
    z_(t-1): Tokenized video representation. Shape [T-1, H', W'] || [B, T-1, H', W'];
                                                T: n. of frames,
                                                H': height / s,
                                                W': width / s;
                                                s is the downscaling factor for VQ-VAE.

    a: Latent action. Shape [T-1, d_a], || [B, T-1, d_a];
                             T-1: n. of actions in all frames T,
                             d_a: dimension of latent embedding (default 32)
    """

    def __init__(
        self,
        vocab_size: int = 8,
        d_model: int = 512,
        d_embedding: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        d_action: int = 16,
        ffn_dim: int = 2048,
    ):
        """
        z_in: [B, T-1, H', W']      for frames 1..T-1
        a: [B, T-1, d_a]            for actions 2..T-1
        """
        super().__init__()

        # Token embeddings
        self.z_embedding = nn.Embedding(vocab_size, d_model)

        # Latent action embeddings
        self.a_embedding = nn.Linear(d_action, d_model)

        # Add ST-Transformer blocks
        self.st_transformer = STTransformer(
            num_heads=num_heads,
            num_layers=num_layers,
            d_model=d_model,
            ffn_dim=d_model * 4,
        )

        # Output projection, d_model -> vocab_size
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(
        self, z: torch.Tensor, a: torch.Tensor, causal_mask: torch.Tensor
    ) -> torch.Tensor:
        # 1. Embeddings
        z_emb = self.z_embedding(z)  # [B, T-1, H', W', dim_model]
        a_emb = self.a_embedding(a)  # [B, T-1, dim_model]

        # 2. "Expand" a as the original action tensor is 1 action per frame only
        # [B, T-1, dim_model] -> [B, T-1, 1, 1, dim_model]
        a_emb = rearrange(a_emb, "b t d -> b t 1 1 d")

        # 3. Combine token and action embeddings (additive)
        x = z_emb + a_emb  # [B, T-1, H', W', dim_model]

        # 4. ST-Transformer with causal mask
        x_transf = self.st_transformer(
            x=x, causal_mask=causal_mask
        )  # [B, T-1, H', W', dim_model]

        # 5. Output Head
        logits = self.output_head(x_transf)  # [B, T-1, H', W', vocab_size]

        return logits
