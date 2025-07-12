import torch
import torch.nn as nn
import torch.nn.functional as F

class NLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood (NLL) loss for regression tasks.
    This loss encourages the model to learn its prediction uncertainty by predicting both mean and variance (or log-variance).
    """
    def __init__(self, reduction: str = 'mean', epsilon: float = 1e-6):
        """
        Initializes the Gaussian Negative Log-Likelihood loss.

        Args:
            reduction (str): Reduction to apply to the output: 'mean' | 'sum' | 'none'. Defaults to 'mean'.
            epsilon (float): Small constant for numerical stability, primarily for `exp(-log_sigma_sq)`.
        """
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Reduction method '{reduction}' not supported.")
        self.reduction = reduction
        self.epsilon = epsilon 

    def forward(self, y_true: torch.Tensor, mu_pred: torch.Tensor, log_sigma_sq_pred: torch.Tensor) -> torch.Tensor:
        """
        Computes the Gaussian Negative Log-Likelihood loss.

        Args:
            y_true (torch.Tensor): True regression targets of shape (N, D), where D is the number of dimensions.
            mu_pred (torch.Tensor): Predicted means of shape (N, D).
            log_sigma_sq_pred (torch.Tensor): Predicted log-variances of shape (N, D).
                                               Using log(sigma^2) ensures positive variance and broader optimization space.

        Returns:
            torch.Tensor: Computed NLL loss.
        """
        if not (y_true.shape == mu_pred.shape == log_sigma_sq_pred.shape):
            raise ValueError("Input tensors y_true, mu_pred, and log_sigma_sq_pred must have the same shape.")

        squared_diff = (y_true - mu_pred).pow(2)
        inv_sigma_sq = torch.exp(-log_sigma_sq_pred)

        loss_term1 = 0.5 * inv_sigma_sq * squared_diff
        loss_term2 = 0.5 * log_sigma_sq_pred
        
        nll_loss_per_element = loss_term1 + loss_term2

        if self.reduction == 'mean':
            return nll_loss_per_element.mean()
        elif self.reduction == 'sum':
            return nll_loss_per_element.sum()
        else: # 'none'
            return nll_loss_per_element

class MSELoss(nn.Module):
    """
    Mean Squared Error (MSE) loss function.
    Common loss for regression tasks, also applicable for uncertainty-weighted pseudo-label regression.
    """
    def __init__(self, reduction: str = 'mean'):
        """
        Initializes the MSE loss.

        Args:
            reduction (str): Reduction to apply to the output: 'mean' | 'sum' | 'none'. Defaults to 'mean'.
        """
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the MSE loss.

        Args:
            inputs (torch.Tensor): Model predictions.
            targets (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Computed MSE loss.
        """
        return self.mse_loss(inputs.float(), targets.float())
