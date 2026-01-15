"""Custom loss functions for OCR training."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCTCLoss(nn.Module):
    """CTC Loss with label smoothing for improved generalization.
    
    Applies label smoothing to the CTC output by mixing the one-hot target
    distribution with a uniform distribution, reducing overconfidence.
    """
    
    def __init__(self, blank: int = 0, smoothing: float = 0.1, zero_infinity: bool = True):
        """
        Args:
            blank: Index of the blank label for CTC.
            smoothing: Label smoothing factor (0.0 = no smoothing, 1.0 = uniform).
            zero_infinity: Whether to zero infinite losses.
        """
        super().__init__()
        self.blank = blank
        self.smoothing = smoothing
        self.ctc_loss = nn.CTCLoss(blank=blank, zero_infinity=zero_infinity, reduction='none')
    
    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            log_probs: Log-softmax output [T, B, C] from the model.
            targets: Target sequences (concatenated).
            input_lengths: Lengths of input sequences.
            target_lengths: Lengths of target sequences.
        
        Returns:
            Smoothed CTC loss (scalar).
        """
        # Standard CTC loss
        ctc_loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        
        # KL divergence term for label smoothing (uniform distribution regularization)
        # log_probs: [T, B, C] -> compute mean negative entropy as smoothing term
        num_classes = log_probs.size(2)
        
        # Uniform distribution has entropy = log(num_classes)
        # KL(uniform || p) = log(num_classes) - H(p) where H(p) = -sum(p * log(p))
        # Since log_probs = log(p), we have: -sum(exp(log_p) * log_p) = -sum(p * log_p)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=2)  # [T, B]
        
        # Average entropy across time and batch
        # We want to maximize entropy (minimize negative entropy)
        smooth_loss = -entropy.mean()
        
        # Combine losses
        total_loss = (1.0 - self.smoothing) * ctc_loss.mean() + self.smoothing * smooth_loss
        
        return total_loss
