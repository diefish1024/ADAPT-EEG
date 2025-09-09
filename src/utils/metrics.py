# src/utils/metrics.py

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import torch
from typing import Dict, Union, Callable

def calculate_classification_metrics(y_true: np.ndarray, y_pred_logits: np.ndarray) -> Dict[str, float]:
    """
    Compute common metrics for classification tasks.

    Args:
        y_true (np.ndarray): True labels, expected shape (N,).
        y_pred_logits (np.ndarray): The raw logits output by the model, expected shape (N, num_classes).

    Returns:
        dict: A dictionary containing accuracy and weighted F1 score.
    """
    # Find the class with the highest predicted probability as the final prediction
    y_pred_labels = np.argmax(y_pred_logits, axis=1)

    accuracy = accuracy_score(y_true, y_pred_labels)
    # 'weighted' F1-score considers the number of samples in each class, suitable for imbalanced datasets.
    f1 = f1_score(y_true, y_pred_labels, average='weighted', zero_division=0) # zero_division=0 handles cases where a class has no true samples or no predicted samples

    return {
        'accuracy': accuracy,
        'f1_score_weighted': f1
    }

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Calculate common metrics for regression tasks.

    Args:
        y_true (np.ndarray): True values, expected shape (N, D) or (N,).
        y_pred (np.ndarray): Predicted values, expected shape (N, D) or (N,).

    Returns:
        dict: Dictionary containing mean squared error (MSE) and R2 score.
              For multi-dimensional regression, also includes R2 per dimension and average R2 across dimensions.
    """
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        'mse': mse,
        'r2': r2
    }

    # For multidimensional regression, calculate R2 for each dimension
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        r2_per_dim = {}
        for i in range(y_true.shape[1]):
            r2_per_dim[f'r2_dim_{i}'] = r2_score(y_true[:, i], y_pred[:, i])
        
        avg_r2_per_dim = np.mean(list(r2_per_dim.values())) # Average across dimensions

        metrics['r2_per_dim'] = r2_per_dim
        metrics['avg_r2_per_dim'] = avg_r2_per_dim
    
    return metrics

def get_metrics_calculator(task_type: str) -> Callable[[torch.Tensor, torch.Tensor], Dict[str, Union[float, Dict[str, float]]]]:
    """
    Returns a callable function that computes metrics based on the specified task type.
    This factory function ensures the correct metric calculation is used for the downstream task.

    Args:
        task_type (str): The type of task, either 'classification' or 'regression'.

    Returns:
        Callable[[torch.Tensor, torch.Tensor], Dict[str, Union[float, Dict[str, float]]]]:
            A function that takes true labels (Tensor) and model outputs (Tensor, e.g., logits or predicted means)
            and returns a dictionary of calculated metrics.

    Raises:
        ValueError: If an unsupported task_type is provided.
    """
    if task_type == 'classification':
        def classification_metrics_wrapper(y_true_tensor: torch.Tensor, y_pred_logits_tensor: torch.Tensor) -> Dict[str, float]:
            """Wrapper for classification metrics, handles tensor to numpy conversion."""
            y_true_np = y_true_tensor.detach().cpu().numpy()
            y_pred_logits_np = y_pred_logits_tensor.detach().cpu().numpy()
            return calculate_classification_metrics(y_true_np, y_pred_logits_np)
        return classification_metrics_wrapper
    elif task_type == 'regression':
        def regression_metrics_wrapper(y_true_tensor: torch.Tensor, y_pred_mu_tensor: torch.Tensor) -> Dict[str, Union[float, Dict[str, float]]]:
            """Wrapper for regression metrics, handles tensor to numpy conversion."""
            y_true_np = y_true_tensor.cpu().numpy()
            y_pred_mu_np = y_pred_mu_tensor.cpu().numpy()
            return calculate_regression_metrics(y_true_np, y_pred_mu_np)
        return regression_metrics_wrapper
    else:
        raise ValueError(f"Unsupported task_type for metrics calculator: '{task_type}'.")

