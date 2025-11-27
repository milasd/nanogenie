import torch
from torch import nn

from nanogenie.models.st_transformer import STTransformer

# Batch size
B = 36
# N. of frames in each sequence
T = 16
D = 8

DIM_EMBEDDINGS = 32


class DynamicsModel(nn.Module):
    """
    Implement the Dynamics Model described in Genie 1.
    It is a decoder-only MaskGIT.

    Input:
    z_(t-1): Tokenized video representation. Shape [T-1, H', W'];
                                                T: n. of frames,
                                                H': height / s,
                                                W': width / s;
                                                s is the downscaling factor for VQ-VAE.

    a: Latent action. Shape [T-1, d_a];
                             T-1: n. of actions in all frames T,
                             d_a: dimension of latent embedding (default 32)
    """

    def __init__(
        self,
        vocab_size: int = 8,
        d_model: int = 512,
        d_embedding: int = DIM_EMBEDDINGS,
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
        self.z_embedding = nn.Embedding(
            vocab_size, d_model
        )  # [B, T-1, H', W'] -> [B, T-1, H', W', d_a], default [36, 16-1, H', W', 32]

        # Latent action embeddings (continuous -> discrete)
        self.a_embedding = nn.Linear(d_action, d_model)

        # Positional embeddings?

        # Add ST-Transformer blocks
        self.st_transformer = STTransformer(
            num_heads=num_heads,
            num_layers=num_layers,
            d_model=d_model,
            ffn_dim=d_model * 4,
        )

        # Output projection, [d_model,] -> [vocab_size]
        self.output_head = nn.Linear(d_model, vocab_size)
