"""
Dynamic models optimizer hyperparameters.
These follow the suggestions from Genie 1 paper
for a smaller experiment to run in a single GPU/TPU.
"""

MAX_LR = 3e-5
MIN_LR = 3e-6
BETA1 = 0.9
BETA2 = 0.9
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 5000
