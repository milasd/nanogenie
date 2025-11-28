import torch

VOCAB_SIZE = 
class MaskGIT:
    @staticmethod
    def apply_masking(z: torch.Tensor, mask_ratio: float, mask_token_id: int = VOCAB_SIZE) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random token masking for MaskGIT training. 

        Args:
            z: Token indices. Shape [B, T-1, H', W']
            mask_ratio: Fraction to mask, from 0.5 to 1.
            mask_token_id: Special ID token for [MASK]

        Return:
            z_masked: Tokens post-masking (some replaced by mask_token_id)
            mask: Mask w/ booleans showing which were masked.
        """
        # Keep frame 1 unmasked; mask 2 to T-1 randomly.


