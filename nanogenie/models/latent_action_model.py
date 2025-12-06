"""
Latent Action Model (LAM) To achieve con-
trollable video generation, we condition each fu-
ture frame prediction on the action taken at the previous frame.
However, such action labels are rarely available in videos from the Internet and ac-
tion annotation can be costly to obtain. Instead, we learn latent actions in a fully unsupervised manner (see Figure 5).
First, an encoder takes as
inputs all previous frames 𝒙1:𝑡 = (𝑥1, · · · 𝑥𝑡) as well as the next frame 𝑥𝑡+1,
and outputs a corresponding set of contin- uous latent actions 𝒂 ̃1:𝑡 = (𝑎 ̃1, · · · 𝑎 ̃𝑡).
A decoder then takes all previous frames and latent actions as input and predicts the next frame 𝑥ˆ𝑡+1."""

"""
    frames [x1...xt]->[ENCODER]->[a1, a...., at-1] -> [DECODER] -> [x_t+1].
                                 [x1, x2,...,xt] _|
    [B, T-1] (actions are transitions between frames, so T-1 actions for T frames)

    
    FIRST BLOCK: ENCODER.
    Input shapes: z_t: [B, T, H, W, C] . Output shape: [B, T-1, Embedding_dim=32]. 

    SECOND BLOCK: DECODER.
    Input shapes: z_t: [B, T, H, W, C], a_t = [B, T-1, Embedding_dim=32]. OUTPUT SHAPE: [B, 1, H, W, C].

"""

import torch

from einops import rearrange
from torch import nn


class LatentActionModel(nn.Module):
    """
    LAM operates directly on raw pixels, not tokenised version.
    """

    def __init__(
        self,
        img_size: tuple[int] = (160, 90),
        patch_scale: int = 16,
        d_model: int = 512,
    ):
        super().__init__()
        # define what's going on here.

        # 1. Patch pixel embeddings.
        self.patches = PatchEmbedding(
            img_size=img_size, patch_scale=patch_scale, d_model=d_model
        )
        # 2. Encoder

        # 3. qnt

        # 4. Decoder


class PatchEmbedding(nn.Module):
    """
    Create embedding of patches from frames x1...xt (raw pixels).

    INPUT:
    [z1...zt]: [B, T, H, W, C] -> [B, T, H // 16, W // 16, C] -> [B, T, H // 16, W // 16, d_embedding=32]

    """

    def __init__(
        self,
        img_size: tuple[int] = (160, 90),
        patch_scale: int = 16,
        d_model: int = 512,
    ):
        super().__init__()
        # Patches of "size" W // 16 x H // 16.
        H, W = img_size
        self.H_patch = H // patch_scale
        self.W_patch = W // patch_scale
        self.n_patches = self.H_patch * self.W_patch

        # Project the raw pixel patches to embedding dim w/ Conv2d.
        self.patch_embedding = nn.Conv2d(
            out_channels=d_model, kernel_size=patch_scale, stride=patch_scale
        )  # [B, T, d_model, H, W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            Raw pixels from frames x. Shpae [B, T, H, W, C]

        Output:
            Embedding of patches of x. Shape [B, T, n_patches, d_model]
        """
        B, T, H, W, C = x.shape

        # 1. rearrange to [B*T, C, H, W] for torch conv2d.
        x_ch = rearrange(x, "b t h w c -> (b t) c h w")  # [B*T, C, H, W]

        # 2. project
        patch_emb = self.patch_embedding(x_ch)  # [B*T, d_model, H', W']

        # 3. Unflatten back and rearrange to [B T (H' W') D]
        patch_emb_rearranged = rearrange(
            patch_emb, "(b t) d h w -> b t (h w) d", b=B, t=T
        )  # Default: [B, T, n_patches, d_model] = default [B, T, 60, 512]

        return patch_emb_rearranged


class LatentActionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # ?
        # ST Transformer block


class LatentAcionDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        pass
