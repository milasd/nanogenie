import torch


class MaskGIT:
    @staticmethod
    def apply_masking(
        z: torch.Tensor, mask_token_id: int = 1024
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random token masking for MaskGIT training.

        Args:
            z: Token indices. Shape [B, T-1, H', W']
            mask_token_id: Special ID token for [MASK]

        Return:
            z_masked: Tokens post-masking (some replaced by mask_token_id)
            mask: Mask w/ booleans showing which were masked. [B, T-1, H', W'], first frame is never masked.
        """
        B, T_minus_1, H, W = z.shape

        # Sample mask ratio: uniform, from 0.5 to 1
        # Sample different ratios for each item in the batch.
        mask_ratio = torch.rand(B, 1, 1, 1, device=z.device) * 0.5 + 0.5  # [B, 1, 1, 1]

        # Create a probability tensor by broadcasting [B, 1, 1, 1] to the full shape
        # Each element in the batch gets its own unique probability p
        probs = mask_ratio.expand(B, T_minus_1 - 1, H, W)

        # Create boolean mask for frames 2 to T-1 (Bernoulli sampling).
        frames_masking = torch.bernoulli(probs).bool()  # [B, T-2, H, W]

        # Create final mask (first element never masked) to match shape of z
        # Define False for first frame (won't mask)
        mask = torch.zeros((B, T_minus_1, H, W), dtype=torch.bool, device=z.device)
        mask[:, 1:, :, :] = frames_masking

        # Apply masking
        z_masked = z.clone()
        z_masked[mask] = mask_token_id  # Replace True pos. w/ mask token

        return z_masked, mask  # [B, T-1, H, W] , [B, T-1, H, W]
