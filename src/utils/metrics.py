# src/utils/metrics.py

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import torch

def calculate_classification_metrics(y_true: torch.Tensor, y_pred_logits: torch.Tensor) -> dict:
    """
    Compute common metrics for classification tasks.

    Args:
        y_true (torch.Tensor): True label, (batch_size,)
        y_pred_logits (torch.Tensor): The raw logits output by the model, (batch_size, num_classes)

    Returns:
        dict: A dictionary containing accuracy and F1 scores.
    """
    # Transfer Tensor to CPU and convert to NumPy array
    y_true_np = y_true.cpu().numpy()
    
    # Find the class with the highest predicted probability as the final prediction
    y_pred_labels_np = torch.argmax(y_pred_logits, dim=1).cpu().numpy()

    accuracy = accuracy_score(y_true_np, y_pred_labels_np)
    # For multi-class classification, F1-score usually requires specifying the average parameter
    # 'weighted' considers the number of samples in each class
    # 'macro' does not consider the number of samples and gives equal importance to all classes
    # 'micro' calculates F1-score globally
    f1 = f1_score(y_true_np, y_pred_labels_np, average='weighted') 

    return {
        'accuracy': accuracy,
        'f1_score_weighted': f1
    }

def calculate_regression_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
    """
    Calculate common metrics for regression tasks.

    Args:
    y_true (torch.Tensor): True value, (batch_size, num_dimensions).
    y_pred (torch.Tensor): Predicted value, (batch_size, num_dimensions).

    Returns:
    dict: Dictionary containing mean squared error (MSE) and R2 score.
    """
    # Transfer Tensor to CPU and convert to NumPy array
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    mse = mean_squared_error(y_true_np, y_pred_np)
    r2 = r2_score(y_true_np, y_pred_np)

    # for multidimensional regression, calculate the R2 and average R2 for each dimension
    if y_true_np.ndim > 1 and y_true_np.shape[1] > 1:
        r2_per_dim = {}
        for i in range(y_true_np.shape[1]):
            r2_per_dim[f'r2_dim_{i}'] = r2_score(y_true_np[:, i], y_pred_np[:, i])
        avg_r2_per_dim = np.mean(list(r2_per_dim.values())) # Average across dimensions
        return {
            'mse': mse,
            'r2': r2,
            'r2_per_dim': r2_per_dim,
            'avg_r2_per_dim': avg_r2_per_dim
        }
    else:
        return {
            'mse': mse,
            'r2': r2
        }
