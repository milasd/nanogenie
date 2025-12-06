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

from einops import rearrange, reduce
from torch import nn

from nanogenie.models.st_transformer import STTransformer


class LatentActionModel(nn.Module):
    """
    Latent Action Model (LAM) implementation following Genie 1 paper description.
    LAM operates directly on raw pixels, not tokenised version.
    Encodes frame pairs into discrete latent actions.

    For frames x={x1...xt}, will output actions a={a1...at-1},
    each a_n referring to an action that happens between a consecutive pair of frames (thus t-1).

    Input:
        Batches of frames (raw pixels) x: torch.Tensor, Shape: [B, T, H, W, C]
    Output:
        Discrete latent actions a: torch.Tensor, Shape [B, T-1, vocab_a]
    """

    def __init__(
        self,
        img_size: tuple[int] = (160, 90),
        patch_scale: int = 16,
        d_model: int = 512,
        vocab_a: int = 6,
    ):
        super().__init__()
        # define what's going on here.

        # 1. Patch pixel embeddings.
        self.patches = PatchEmbedding(
            img_size=img_size, patch_scale=patch_scale, d_model=d_model
        )  # [B, T, n_patches, d_model]

        # 2. Encoder
        self.encoder = LatentActionEncoder()

        # 3. qnt

        # 4. Decoder
        self.decoder = LatentAcionDecoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Patch embeddings
        patch_embeddings = self.patches(x)

        # 2. Encoder


class PatchEmbedding(nn.Module):
    """
    Create embedding of patches from frames x1...xt (raw pixels).

    INPUT:
    x: {x1...xt}: [B, T, H, W, C] -> [B, T, H // 16, W // 16, C] ->
                -> [B, T, H // 16, W // 16, d_model=512] -> [B, T, n_patches, d_model]

    Output:
    x_emb: torch.Tensor. Embeddings of patches, Shape [B, T, n_patches, d_model]
    """

    def __init__(
        self,
        img_size: tuple[int] = (160, 90),
        patch_scale: int = 16,
        d_model: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        num_codes: int = 6,
        latent_dim: int = 32,
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
            Embedding of patches of x. Shape [B, T, H_patch, W_patch, d_model]
        """
        B, T, H, W, C = x.shape

        # 1. rearrange to [B*T, C, H, W] for torch conv2d.
        x_ch = rearrange(x, "b t h w c -> (b t) c h w")  # [B*T, C, H, W]

        # 2. project
        patch_emb = self.patch_embedding(x_ch)  # [B*T, d_model, H', W']

        # 3. Unflatten back and rearrange to [B T H' W' D]
        patch_emb_rearranged = rearrange(
            patch_emb, "(b t) d h w -> b t h w d", b=B, t=T
        )  # Default: [B, T, n_patches, d_model] = default [B, T, H', W', 512]

        return patch_emb_rearranged


class LatentActionEncoder(nn.Module):
    """Encode frame pairs into latent action representation."""

    def __init__(
        self,
        num_layers: int = 8,
        num_heads: int = 8,
        d_model: int = 512,
        latent_dim: int = 32,
    ):
        super().__init__()

        # ST Transformer
        self.st_transformer = STTransformer(
            num_layers=num_layers, num_heads=num_heads, d_model=d_model
        )

        # Project [..., d_model] -> [..., latent_dim]
        self.action_head = nn.Linear(d_model, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the patches from the frames.
        """
        # 1. Pass patches through transformer. [B, T, H', W', d_model]
        encoded = self.st_transformer(x)

        # 2. Spatial feature aggregation. Try mean pooling?
        frame_features = reduce(
            encoded, "b t h w d -> b t d", "mean"
        )  # [B, T, d_model]

        # 3. Temporal slicing [T -> T-1]
        # This works as frame 1 already contains "information" from 0
        # Due to ST transformer causal temporal attention.
        frame_tmp = frame_features[:, 1:, :]  # [B, T-1, d_model]

        # 4. Project into latent_dim: [.., d_model] -> [..., latent_dim]
        a_continuous = self.action_head(frame_tmp)

        return a_continuous  # [B, T-1, latent_dim]


class LatentAcionDecoder(nn.Module):
    """Decode latent action to reconstruct frame x_t+1."""

    def __init__(self):
        super().__init__()
        pass
