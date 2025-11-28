"""
This file contains the shapes and hyperparameters etc.
of the models, following the simple implementation in the Genie paper.
"""

# ------------------------

# Frame/Video dimensions

T = 16  # Sequence length (frames)
H = 64  # Frame height (will likely change for the reduced implementation)
W = 64  # Frame width  (will likely change for the reduced implementation)
C = 3  # Color channels


# ------------------------

# Video tokenizer (z)

P_z = 4  # Patch size for tokenizer
H_z = H // P_z  # = 16 (height in tokens)
W_z = W // P_z  # = 16 (width in tokens)
D_z = H_z * W_z  # = 256 (spatial tokens per frame, called D in paper)
V_z = 1024  # Codebook size (number of unique codes)
E_z = 32  # Embedding/latent dimension for codebook

# ------------------------

# Latent action model (a)

P_a = 16  # Patch size for LAM
H_a = H // P_a  # ≈ 4 (height in tokens)
W_a = W // P_a  # = 4 (width in tokens)
D_a = H_a * W_a  # ≈ 16 (spatial tokens per frame for LAM)
V_a = 6  # Number of latent actions (codebook size, |A| in paper, 8 for main exp. but 6 for reduced)
E_a = 32  # Action embedding dimension

# Architecture

D_lam = 512  # d_model
L_lam = 8  # num_layers
N_heads_lam = 8  # num_heads


# For the simple experiment, B is likely 48


# ----------------

# Dynamics model

D_dyn = 512  # d_model
L_dyn = 12  # num_layers
N_heads_dyn = 8  # num_heads

# MaskGIT sampling
maskgit_steps = 25  # Number of MaskGIT steps per frame
temperature = 1.0  # Sampling temperature (lower than main model)

# Training
B_dyn = 36  # Batch size (sequences)
# Total images per batch = B_dyn * T = 36 * 16 = 576
