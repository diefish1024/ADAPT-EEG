# src/utils/metrics.py

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import torch

def calculate_classification_metrics(y_true: torch.Tensor, y_pred_logits: torch.Tensor) -> dict:
    """
    计算分类任务的常用指标。

    Args:
        y_true (torch.Tensor): 真实标签，形状 (batch_size,)。
        y_pred_logits (torch.Tensor): 模型输出的原始 logits ，形状 (batch_size, num_classes)。

    Returns:
        dict: 包含准确率和 F1 分数的字典。
    """
    # 将 Tensor 转移到 CPU 并转换为 NumPy 数组
    y_true_np = y_true.cpu().numpy()
    
    # 找到预测概率最高的类别作为最终预测
    y_pred_labels_np = torch.argmax(y_pred_logits, dim=1).cpu().numpy()

    accuracy = accuracy_score(y_true_np, y_pred_labels_np)
    # 对于多类别分类，F1-score 通常需要指定 average 参数
    # 'weighted' 考虑每个类别的样本数量
    # 'macro' 不考虑样本数量，对所有类别的重要性相同
    # 'micro' 全局计算 F1-score
    f1 = f1_score(y_true_np, y_pred_labels_np, average='weighted') 

    return {
        'accuracy': accuracy,
        'f1_score_weighted': f1
    }

def calculate_regression_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
    """
    计算回归任务的常用指标。

    Args:
        y_true (torch.Tensor): 真实值，形状 (batch_size, num_dimensions)。
        y_pred (torch.Tensor): 预测值，形状 (batch_size, num_dimensions)。

    Returns:
        dict: 包含均方误差 (MSE) 和 R2 分数的字典。
    """
    # 将 Tensor 转移到 CPU 并转换为 NumPy 数组
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    mse = mean_squared_error(y_true_np, y_pred_np)
    r2 = r2_score(y_true_np, y_pred_np)

    # 如果是多维回归，计算每个维度的R2和平均R2
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
