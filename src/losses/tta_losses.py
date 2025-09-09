# src/losses/tta_losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict, Literal

class PICLoss(nn.Module):
    """
    Prediction-Informed Clustering (PIC) Loss.
    This loss function promotes clustering of features based on soft pseudo-labels,
    encouraging intra-class compactness and inter-class separability.
    It is designed to be scale-invariant, preventing trivial solutions.
    Ref: https://arxiv.org/abs/2410.06976
    Args:
        temperature (float): A temperature parameter for scaling logits before softmax,
                             which can influence the sharpness of pseudo-labels.
                             (Not used in current implementation, but can be added if needed)
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        # We keep it for future flexibility.
        self.temperature = temperature
    
    def forward(self, feats: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Prediction-Informed Clustering (PIC) loss.
        Args:
            feats (torch.Tensor): Node/Graph features from the feature extractor,
                                  shape (batch_size, feature_dim).
            prob (torch.Tensor): Soft predictions (pseudo-labels) from the base TTA method,
                                 shape (batch_size, num_classes).
                                 It's expected to be detached to prevent gradients cycling back.
        Returns:
            torch.Tensor: The calculated PIC loss.
        """
        if feats.dim() != 2 or prob.dim() != 2:
            raise ValueError("feats and prob must be 2-dimensional tensors (batch_size, feature_dim/num_classes).")
        if feats.shape[0] != prob.shape[0]:
            raise ValueError("Batch sizes of feats and prob must match.")
        num_classes = prob.shape[1]
        
        sum_prob_per_class = prob.sum(dim=0).view(num_classes, 1)
        
        # Calculate centroids (means) for each pseudo-class
        # (num_classes, feature_dim) = (num_classes, batch_size) @ (batch_size, feature_dim) / (num_classes, 1)
        means = torch.zeros(num_classes, feats.shape[1], device=feats.device, dtype=feats.dtype)
        valid_classes_mask = sum_prob_per_class.squeeze() > 1e-6 # More robust check
        
        if valid_classes_mask.any():
            means[valid_classes_mask] = (prob[:, valid_classes_mask].T @ feats) / sum_prob_per_class[valid_classes_mask]
        # Calculate squared distance from each feature to each centroid: (batch_size, num_classes)
        sq_dist_to_means = torch.square(torch.cdist(feats, means, p=2))
        # Calculate intra-class variance: Sigma_intra^2 = sum_i sum_c prob_ic * ||z_i - mu_c||^2
        # (batch_size, num_classes) * (batch_size, num_classes) --> sum over all elements
        var_intra = (prob * sq_dist_to_means).sum()
        # Calculate total variance: Sigma_total^2 = sum_i ||z_i - mu_*||^2
        # where mu_* is the global mean of features
        global_mean = feats.mean(dim=0) # (feature_dim,)
        var_total = torch.sum(torch.square(feats - global_mean))
        # The PIC loss is the ratio of intra-class variance to total variance
        loss = var_intra / (var_total + 1e-8) 
        return loss

class EntropyMinimizationLoss(nn.Module):
    """
    Entropy Minimization Loss, commonly used for Test-Time Adaptation (TTA).
    Aims to make the model produce high-confidence predictions on target domain data by reducing prediction entropy.
 
    For classification, it minimizes the Shannon entropy of the predicted probability distribution.
    For regression, it minimizes the entropy of the predicted Gaussian distribution, which is
    equivalent to minimizing its variance (log_sigma_sq).
    """
    def __init__(self, task_type: Literal['classification', 'regression'], reduction: str = 'mean'):
        """
        Initializes the Entropy Minimization Loss.
 
        Args:
            task_type (Literal['classification', 'regression']): Specifies the type of task
                                                          this loss will be applied to.
            reduction (str): Specifies the reduction to apply: 'mean' | 'sum' | 'none'. Defaults to 'mean'.
        """
        super().__init__()
        if task_type not in ['classification', 'regression']:
            raise ValueError(f"Unsupported task_type: '{task_type}'. Must be 'classification' or 'regression'.")
        self.task_type = task_type
 
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Reduction method '{reduction}' not supported.")
        self.reduction = reduction
 
    def _calculate_classification_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Calculates the entropy for classification logits.
        H(p) = - sum(p * log p)
 
        Args:
            logits (torch.Tensor): Raw model outputs (logits) before softmax. Shape (N, C).
 
        Returns:
            torch.Tensor: Entropy per sample. Shape (N,).
        """
        prob = F.softmax(logits, dim=-1)
        log_prob = F.log_softmax(logits, dim=-1)
        
        # Calculate entropy: -sum(p * log(p)).
        # torch.log_softmax handles numerical stability for log(0) cases by returning -inf.
        # When multiplied by prob (which is 0 in that case), it effectively becomes 0.
        entropy_per_sample = -(prob * log_prob).sum(dim=-1)
        return entropy_per_sample
 
    def _calculate_regression_uncertainty_minimization(self, log_sigma_sq: torch.Tensor) -> torch.Tensor:
        """
        Calculates a term for uncertainty minimization in regression, equivalent to minimizing Gaussian entropy.
        For a Gaussian distribution, H(X) = 0.5 * log(2 * pi * e * sigma^2).
        Minimizing H(X) is equivalent to minimizing log(sigma^2) (assuming other terms are constant).
        This function directly returns log_sigma_sq, which when minimized, reduces uncertainty.
 
        Args:
            log_sigma_sq (torch.Tensor): Model's predicted log-variance. Shape (N, D).
                                         N is batch size, D is number of regression dimensions.
 
        Returns:
            torch.Tensor: log-variance per sample, averaged across regression dimensions if D > 1.
                          If D=1, shape is (N,). If D>1 and reduction='mean' in forward,
                          it will be averaged across (N*D) elements.
        """
        # For multi-dimensional regression, the total entropy of N independent Gaussians
        # is the sum of their individual entropies. Minimizing the sum of log_sigma_sq
        # across dimensions is equivalent to minimizing total entropy.
        # Here we return the log_sigma_sq as is, and the 'reduction' in forward will handle
        # whether to sum across dimensions and/or batch.
        return log_sigma_sq # Shape (N, D) or (N,)
 
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Computes the entropy minimization loss based on the initialized task_type.
 
        Args:
            inputs (torch.Tensor):
                - If task_type is 'classification': Raw classification logits of shape (N, C).
                - If task_type is 'regression': Predicted log_sigma_sq of shape (N, D).
 
        Returns:
            torch.Tensor: Computed entropy minimization loss after applying the specified reduction.
        """
        if self.task_type == 'classification':
            per_sample_term = self._calculate_classification_entropy(inputs) # Shape (N,)
        elif self.task_type == 'regression':
            per_sample_term = self._calculate_regression_uncertainty_minimization(inputs) # Shape (N, D) or (N,)
        
        # Apply reduction. Note: for regression, if per_sample_term is (N, D),
        # 'mean' reduction will average over N*D elements.
        if self.reduction == 'mean':
            return per_sample_term.mean()
        elif self.reduction == 'sum':
            return per_sample_term.sum()
        else: # 'none'
            return per_sample_term

class UncertaintyWeightedConsistencyLoss(nn.Module):
    """
    Uncertainty-Weighted Consistency Regularization Loss for regression tasks.
    Penalizes deviations between predictions of different augmented versions of the same input.
    Weights are inversely proportional to predicted variances, promoting consistent and high-confidence predictions.
    """
    def __init__(self, reduction: str = 'mean', epsilon: float = 1e-6):
        """
        Initializes the Uncertainty-Weighted Consistency Regularization Loss.
 
        Args:
            reduction (str): Specifies the reduction to apply: 'mean' | 'sum' | 'none'. Defaults to 'mean'.
            epsilon (float): Small constant for numerical stability to prevent division by zero.
        """
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Reduction method '{reduction}' not supported.")
        self.reduction = reduction
        self.epsilon = epsilon
 
    def forward(self,
                mu_pred1: torch.Tensor, log_sigma_sq_pred1: torch.Tensor,
                mu_pred2: torch.Tensor, log_sigma_sq_pred2: torch.Tensor) -> torch.Tensor:
        """
        Computes the uncertainty-weighted consistency loss.
 
        Args:
            mu_pred1 (torch.Tensor): Predicted means for the first augmented version (N, D).
            log_sigma_sq_pred1 (torch.Tensor): Predicted log-variances for the first augmented version (N, D).
            mu_pred2 (torch.Tensor): Predicted means for the second augmented version (N, D).
            log_sigma_sq_pred2 (torch.Tensor): Predicted log-variances for the second augmented version (N, D).
 
        Returns:
            torch.Tensor: Computed uncertainty-weighted consistency loss.
        """
        if not (mu_pred1.shape == log_sigma_sq_pred1.shape == mu_pred2.shape == log_sigma_sq_pred2.shape):
            raise ValueError("All input tensors for consistency loss must have the same shape.")
 
        sigma_sq_pred1 = torch.exp(log_sigma_sq_pred1)
        sigma_sq_pred2 = torch.exp(log_sigma_sq_pred2)
 
        # Weights: 1 / (sigma_1^2 + sigma_2^2 + epsilon)
        weights = 1.0 / (sigma_sq_pred1 + sigma_sq_pred2 + self.epsilon)
 
        # Squared L2 norm between means, summed over feature dimensions (D)
        diff_mu_squared = (mu_pred1 - mu_pred2).pow(2).sum(dim=-1) # (N,)
 
        # Per-sample consistency loss: mean of weights (over D) * squared diff (summed over D)
        # Assuming weights should be averaged over dimensions if D > 1 for a single per-sample weight.
        # If D is typically 1 (e.g., scalars), or if each D dimension is independent and weighted separately,
        # then `.mean(dim=-1)` might not be suitable if it implies averaging the weight.
        # It's more common to have (N, D) -> (N,) for weights to match diff_mu_squared.
        # Correct approach for per-sample loss: Mean of (weighted squared_diff) over dimensions.
        consistency_loss_per_element = weights * (mu_pred1 - mu_pred2).pow(2)
        
        if self.reduction == 'mean':
            return consistency_loss_per_element.mean()
        elif self.reduction == 'sum':
            return consistency_loss_per_element.sum()
        else: # 'none'
            return consistency_loss_per_element

class ConsistencyLoss(nn.Module):
    """
    Standard consistency loss.
    Applicable for classification (logits/probabilities) or regression (mu_pred).
    Encourages the model to output consistent predictions when faced with different perturbations of the same input.
    """
    def __init__(self, reduction: str = 'mean', loss_type: str = 'mse', temp: float = 1.0):
        """
        Initializes the consistency loss.
 
        Args:
            reduction (str): Specifies the reduction to apply to the output: 'mean' | 'sum' | 'none'. Defaults to 'mean'.
            loss_type (str): Internal loss type for measuring consistency:
                             'mse' (for regression values, classification logits, or features) |
                             'kl_div' (for classification probability distributions).
            temp (float): Temperature parameter, used for KL divergence with softmax.
        """
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Reduction method '{reduction}' not supported.")
        if loss_type not in ['mse', 'kl_div']:
            raise ValueError(f"Loss type '{loss_type}' not supported. Must be 'mse' or 'kl_div'.")
        self.reduction = reduction
        self.loss_type = loss_type
        self.temp = temp
 
    def forward(self, pred1: torch.Tensor, pred2: torch.Tensor) -> torch.Tensor:
        """
        Computes the consistency loss.
 
        Args:
            pred1 (torch.Tensor): First predicted output (logits, probabilities, or regression values).
            pred2 (torch.Tensor): Second predicted output (logits, probabilities, or regression values).
 
        Returns:
            torch.Tensor: Computed consistency loss.
        """
        if pred1.shape != pred2.shape:
            raise ValueError("Input tensors for consistency loss must have the same shape.")
 
        if self.loss_type == 'mse':
            # Applicable for regression values, classification logits, or features
            # F.mse_loss with reduction='none' computes element-wise squared error.
            # We then sum over the last dimension (e.g., feature/class dimension) to get per-sample loss.
            consistency_loss_per_sample = F.mse_loss(pred1, pred2, reduction='none').sum(dim=-1) # (N,)
        elif self.loss_type == 'kl_div':
            # Applicable for classification probability distributions (logits)
            # D_KL(P || Q) where P is prob2 and Q is prob1
            # F.kl_div(log_Q, P, ...)
            # log_softmax is used for numerical stability
            
            # log_prob1 = log(Q)
            log_prob1 = F.log_softmax(pred1 / self.temp, dim=-1)
            # prob2 = P
            prob2 = F.softmax(pred2 / self.temp, dim=-1) # Often, targets are often detached and stop gradients here.
 
            # F.kl_div(input (log P), target (P)) computes D_KL(target || exp(input))
            # So, to compute D_KL(prob2 || prob1), we need F.kl_div(log_prob1, prob2)
            # With reduction='batchmean', it internally sums and divides by batch size.
            # If reduction='none', it gives per-element KL and we sum over dims.
            
            # Using reduction='none' for per-sample calculation then sum over class dimension
            kl_div_per_element = F.kl_div(log_prob1, prob2, reduction='none', log_target=False)
            consistency_loss_per_sample = kl_div_per_element.sum(dim=-1) # (N,)
 
        if self.reduction == 'mean':
            return consistency_loss_per_sample.mean()
        elif self.reduction == 'sum':
            return consistency_loss_per_sample.sum()
        else: # 'none'
            return consistency_loss_per_sample

class InfoNCELoss(nn.Module):
    """
    Contrastive learning loss function (InfoNCE).
    Optimizes feature representations for discriminability and domain invariance
    by pulling together different augmented views of the same sample in the latent space,
    and pushing apart different (negative) samples.
    """
    def __init__(self, temperature: float = 0.5, reduction: str = 'mean'):
        """
        Initializes the InfoNCE loss.
 
        Args:
            temperature (float): Temperature parameter to scale the similarity distribution.
            reduction (str): Specifies the reduction to apply to the output: 'mean' | 'sum' | 'none'. Defaults to 'mean'.
        """
        super().__init__()
        self.temperature = temperature
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Reduction method '{reduction}' not supported.")
        self.reduction = reduction
 
    def forward(self, queries: torch.Tensor, keys: torch.Tensor, # queries and keys could be features from different views
                 negative_keys: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes the InfoNCE loss.
 
        Args:
            queries (torch.Tensor): Anchor features (N, D).
            keys (torch.Tensor): Positive sample features (N, D).
            negative_keys (torch.Tensor, optional): Negative sample features (K, D).
                                                    If None, other samples within the batch are used as negatives.
 
        Returns:
            torch.Tensor: Computed InfoNCE loss.
        """
        # Normalize features
        queries = F.normalize(queries, dim=-1)
        keys = F.normalize(keys, dim=-1)
 
        # Positive pair similarity: sim(q_i, k_i) -> (N, 1)
        l_pos = torch.einsum('nc,nc->n', [queries, keys]).unsqueeze(-1) # Shape (N, 1)
 
        # Negative pair similarities
        if negative_keys is None:
            batch_sim_matrix = torch.einsum('nc,kc->nk', [queries, keys]) # Shape (N, N)
            
            neg_mask = ~torch.eye(batch_sim_matrix.shape[0], dtype=torch.bool, device=batch_sim_matrix.device)
            
            l_neg_list = []
            for i in range(batch_sim_matrix.shape[0]):
                l_neg_list.append(batch_sim_matrix[i, neg_mask[i]])
            l_neg = torch.stack(l_neg_list, dim=0) # Shape (N, N-1)
        else:
            negative_keys = F.normalize(negative_keys, dim=-1)
            l_neg = torch.einsum('nc,kc->nk', [queries, negative_keys.transpose(0, 1)]) # Shape (N, K)
 
        # Concatenate positive and negative similarities
        logits = torch.cat([l_pos, l_neg], dim=1) # Shape (N, 1 + num_negatives)
        
        # Scale by temperature
        logits /= self.temperature
 
        # Labels: the positive sample is at index 0 for each query
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
 
        # Compute cross-entropy loss per sample
        loss_per_sample = F.cross_entropy(logits, labels, reduction='none')
 
        if self.reduction == 'mean':
            return loss_per_sample.mean()
        elif self.reduction == 'sum':
            return loss_per_sample.sum()
        else: # 'none'
            return loss_per_sample

class DomainAdversarialLoss(nn.Module):
    """
    Domain Adversarial Neural Network (DANN) loss, comprising the domain discriminator loss
    and the encoder's adversarial loss. The encoder aims to fool the discriminator
    into classifying features as domain-invariant, typically achieved via a Gradient Reversal Layer (GRL).
    """
    def __init__(self, reduction: str = 'mean', lambda_tradeoff: float = 1.0):
        """
        Initializes the Domain Adversarial Loss.
 
        Args:
            reduction (str): Specifies the reduction to apply to the output: 'mean' | 'sum' | 'none'. Default is 'mean'.
            lambda_tradeoff (float): Weight for the encoder's adversarial loss contribution to the total loss.
        """
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=reduction)
        self.reduction = reduction
        self.lambda_tradeoff = lambda_tradeoff
 
    def forward(self,
                source_features_proj: torch.Tensor,
                target_features_proj: torch.Tensor,
                domain_discriminator: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the DANN loss.
 
        Args:
            source_features_proj (torch.Tensor): Projected features from the source domain.
            target_features_proj (torch.Tensor): Projected features from the target domain.
            domain_discriminator (nn.Module): The domain discriminator model.
 
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - discriminator_loss (torch.Tensor): Loss for the domain discriminator.
                - encoder_adversarial_loss (torch.Tensor): Adversarial loss for the feature encoder.
        """
        # Ensure inputs are at least 2D (batch_size, feature_dim)
        if source_features_proj.dim() == 1:
            source_features_proj = source_features_proj.unsqueeze(0)
        if target_features_proj.dim() == 1:
            target_features_proj = target_features_proj.unsqueeze(0)
 
        # 1. Domain Discriminator Loss (L_D)
        # The discriminator aims to correctly classify source and target domain features.
        source_domain_preds = domain_discriminator(source_features_proj)
        target_domain_preds = domain_discriminator(target_features_proj)
 
        # Source domain labels are 1 (real/source), target domain labels are 0 (fake/target)
        source_labels = torch.ones_like(source_domain_preds) 
        target_labels = torch.zeros_like(target_domain_preds) 
 
        discriminator_loss_source = self.bce_loss(source_domain_preds, source_labels)
        discriminator_loss_target = self.bce_loss(target_domain_preds, target_labels)
        discriminator_loss = discriminator_loss_source + discriminator_loss_target
 
        # 2. Encoder Adversarial Loss (L_E_adv)
        # The encoder aims to fool the discriminator to learn domain-invariant features.
        # This is where the Gradient Reversal Layer (GRL) plays a crucial role.
        # The encoder wants *target* domain features to be classified as *source* domain features (label 1).
        encoder_adversarial_labels = torch.ones_like(target_domain_preds)
        encoder_adversarial_loss = self.bce_loss(target_domain_preds, encoder_adversarial_labels)
 
        # NOTE on GRL placement:
        # The GRL must be inserted between the feature extractor's output and the domain discriminator's input.
        # This implementation assumes GRL is applied to target_features_proj *before* it enters `domain_discriminator`
        # for encoder_adversarial_loss calculation, or that discriminator itself handles it.
        
        return discriminator_loss, self.lambda_tradeoff * encoder_adversarial_loss

class WeightedCORALLoss(nn.Module):
    """
    Weighted CORAL (CORrelation ALignment) Loss.
    Aligns the weighted covariance matrices of source and target domains in a salient subspace
    to minimize domain discrepancy.
    """
    def __init__(self, reduction: str = 'mean'):
        """
        Initializes the Weighted CORAL Loss.
 
        Args:
            reduction (str): Specifies the reduction to apply to the output: 'mean' | 'sum' | 'none'. Default is 'mean'.
        """
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Reduction method '{reduction}' not supported.")
        self.reduction = reduction
 
    def forward(self,
                source_features_proj: torch.Tensor,
                target_features_proj: torch.Tensor,
                source_eigenvalues: torch.Tensor, # lambda_s: (K,)
                dimension_weights: torch.Tensor, # a_d: (K,)
                epsilon: float = 1e-5 # Numerical stability for covariance matrix computation
                ) -> torch.Tensor:
        """
        Computes the Weighted CORAL Loss.
 
        Args:
            source_features_proj (torch.Tensor): Projected features from the source domain (N_S, K).
                                                 These are features projected into the salient subspace.
            target_features_proj (torch.Tensor): Projected features from the target domain (N_T, K).
                                                 These are features projected into the salient subspace.
            source_eigenvalues (torch.Tensor): Eigenvalues of the source domain feature covariance matrix, shape (K,).
                                               This represents the diagonal form of the source covariance matrix (Lambda^s).
            dimension_weights (torch.Tensor): Weighting factors for each dimension, shape (K,).
                                              Represented as a diagonal matrix A = diag(a_1, ..., a_K).
            epsilon (float): A small value for numerical stability, especially when calculating covariance.
 
        Returns:
            torch.Tensor: The computed Weighted CORAL loss.
        """
        # Ensure inputs are at least 2D (batch_size, feature_dim)
        if source_features_proj.dim() == 1:
            source_features_proj = source_features_proj.unsqueeze(0)
        if target_features_proj.dim() == 1:
            target_features_proj = target_features_proj.unsqueeze(0)
 
        # Get feature dimension K
        if source_features_proj.size(1) != target_features_proj.size(1):
            raise ValueError("Source and target projected features must have the same feature dimension K.")
        K = source_features_proj.size(1)
 
        # Validate shapes of eigenvalues and weights
        if source_eigenvalues.shape != (K,) or dimension_weights.shape != (K,):
            raise ValueError(f"source_eigenvalues and dimension_weights must have shape ({K},).")
 
 
        def _compute_covariance(features: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
            """
            Computes the unbiased covariance matrix for a batch of features.
            C = (features - mean_f).T @ (features - mean_f) / (N - 1)
            
            Args:
                features (torch.Tensor): Input features of shape (N, K).
                eps (float): Small value added to the diagonal for numerical stability.
 
            Returns:
                torch.Tensor: The covariance matrix of shape (K, K).
            """
            N = features.size(0)
            if N <= 1:
                # If batch size is too small, covariance cannot be reliably computed.
                # Returning a zero matrix or raising an error are common practices.
                # For loss calculation, returning zeros means no loss contribution from this term.
                return torch.zeros(K, K, device=features.device, dtype=features.dtype)
 
            mean_f = torch.mean(features, dim=0, keepdim=True)
            centered_f = features - mean_f
            
            # Compute covariance matrix: (K, N) @ (N, K) -> (K, K)
            covariance_matrix = (centered_f.T @ centered_f) / (N - 1)
            
            # Add epsilon to the diagonal for numerical stability.
            covariance_matrix = covariance_matrix + torch.eye(K, device=features.device, dtype=features.dtype) * eps
            return covariance_matrix
 
        # Compute target domain covariance matrix C_T
        C_T = _compute_covariance(target_features_proj, self.epsilon)
 
        # Construct Weighted Source Covariance Matrix (A_diag_matrix @ Lambda_s_diag_matrix).
        # Lambda_s is a diagonal matrix whose diagonal elements are `source_eigenvalues`.
        # A is a diagonal matrix whose diagonal elements are `dimension_weights`.
        # The result (A @ Lambda_s) is still a diagonal matrix with diagonal elements `a_d * lambda_d^s`.
        weighted_source_cov = torch.diag(dimension_weights * source_eigenvalues)
 
        # Construct A_diag_matrix from dimension_weights for the Hadamard product on C_T.
        A_diag_matrix = torch.diag(dimension_weights)
        
        # Compute Weighted Target Covariance Matrix (A_diag_matrix Hadamard C_T).
        # This operation means element-wise product. If A_diag_matrix is indeed diagonal,
        # then (A_diag_matrix * C_T)_{ij} = A_diag_matrix_{ii} * C_T_{ij} for diagonal elements of A_diag_matrix.
        # This effectively scales each row of C_T by the corresponding dimension weight.
        weighted_target_cov = A_diag_matrix * C_T # Hadamard product
 
        # Calculate the Frobenius norm squared of the difference between the
        # weighted source and weighted target covariance matrices.
        # || (A * Lambda^s) - (A * C_T) ||_F^2
        diff_matrix = weighted_source_cov - weighted_target_cov
        loss = torch.linalg.norm(diff_matrix, ord='fro')**2
 
        # Apply reduction. For a single scalar loss, 'mean', 'sum', 'none'
        # will yield the same scalar value, but 'none' would typically return a (1,) tensor.
        if self.reduction == 'mean':
            return loss
        elif self.reduction == 'sum':
            return loss
        else: # 'none'
            return loss.unsqueeze(0) # Ensure it's a tensor of shape (1,)


class UncertaintyWeightedPseudoLabelLoss(nn.Module):
    """
    Uncertainty-Weighted Pseudo-Label Loss.
    Utilizes high-confidence predictions (low variance for regression, high probability for classification)
    as pseudo-labels. Applicable to both regression and classification tasks.
    """
    def __init__(self,
                 confidence_thresholds: Union[float, Dict[str, float]],
                 reduction: str = 'mean',
                 epsilon: float = 1e-6):
        """
        Initializes the Uncertainty-Weighted Pseudo-Label Loss.
 
        Args:
            confidence_thresholds (Union[float, Dict[str, float]]): Threshold(s) for determining high-confidence samples.
                - If float: For regression, this is the variance threshold (samples below this are high-confidence).
                            For classification, this is the maximum prediction probability threshold (samples above this are high-confidence).
                - If Dict[str, float]: Dictionary form, e.g., {'regression': 0.1, 'classification': 0.9},
                                       allowing different thresholds for different tasks.
            reduction (str): Specifies the reduction to apply to the output: 'mean' | 'sum' | 'none'.
            epsilon (float): Small value for numerical stability in weight calculation.
        """
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Reduction method '{reduction}' not supported.")
        self.reduction = reduction
        self.epsilon = epsilon
        # For regression: computes element-wise MSE.
        self.mse_loss = nn.MSELoss(reduction='none') 
 
        if isinstance(confidence_thresholds, dict):
            if 'regression' not in confidence_thresholds and 'classification' not in confidence_thresholds:
                raise ValueError("If 'confidence_thresholds' is a dict, it must contain at least 'regression' or 'classification' keys.")
            self.regression_threshold = confidence_thresholds.get('regression')
            self.classification_threshold = confidence_thresholds.get('classification')
        else:
            self.regression_threshold = confidence_thresholds
            self.classification_threshold = confidence_thresholds
 
    def forward(self,
                predictions: torch.Tensor,       # For regression: mu_pred (N, D). For classification: logits (N, C)
                uncertainties: Optional[torch.Tensor] = None, # For regression: log_sigma_sq_pred (N, D). For classification: None
                task_type: str = 'regression'
               ) -> torch.Tensor:
        """
        Computes the uncertainty-weighted pseudo-label loss.
 
        Args:
            predictions (torch.Tensor):
                - For 'regression': Predicted mean of affective dimensions `mu_pred` (N, D).
                - For 'classification': Raw predicted logits `logits` (N, C).
            uncertainties (Optional[torch.Tensor]):
                - For 'regression': Predicted log-variance `log_sigma_sq_pred` (N, D).
                - For 'classification': Should be `None`.
            task_type (str): Specifies the task type, 'regression' or 'classification'.
 
        Returns:
            torch.Tensor: The computed pseudo-label loss.
        """
        # Default loss is 0.0, ensures backprop consistency even if no confident samples.
        total_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True) 
 
        if task_type == 'regression':
            if uncertainties is None:
                raise ValueError("For 'regression' task_type, 'uncertainties' (log_sigma_sq_pred) must be provided.")
            
            mu_pred = predictions
            log_sigma_sq_pred = uncertainties
            sigma_sq_pred = torch.exp(log_sigma_sq_pred)
 
            # Pseudo-labels are the model's own predicted means (detached from computation graph).
            # NOTE: As previously discussed, an expression like (mu_pred - mu_pred.detach()) has a zero gradient
            # with respect to mu_pred, meaning mu_pred will not be updated by this loss term.
            # This specific formulation would typically be used to regularize the sigma_sq_pred,
            # or it might be intended for scenarios where `pseudo_labels` come from a distinct source (e.g., EMA historical model,
            # or another augmented view of the same input), which is not the case here.
            pseudo_labels = mu_pred.detach()
 
            # Calculate element-wise MSE loss: (N, D)
            # This computes ||mu_pred - pseudo_labels||^2
            per_element_mse = self.mse_loss(mu_pred, pseudo_labels)
 
            # Calculate weights: 1 / (sigma_sq + epsilon).
            # Lower variance (higher confidence) leads to a larger weight.
            # weights.shape: (N, D)
            weights = 1.0 / (sigma_sq_pred + self.epsilon)
 
            # Weighted loss components: (N, D)
            # This implements (||y_pseudo,j - mu_j||^2) / (sigma_j^2)
            weighted_components = per_element_mse * weights
 
            # Filter samples based on confidence threshold
            if self.regression_threshold is not None:
                # Create a mask for confident samples: all dimensions' variances must be below the threshold.
                confident_mask = (sigma_sq_pred < self.regression_threshold).all(dim=-1) # (N,)
                
                if confident_mask.sum() == 0:
                    # If no confident samples, return zero loss immediately.
                    return total_loss
                
                # Filter out loss components for high-confidence samples.
                filtered_loss = weighted_components[confident_mask] # (N_confident, D)
            else:
                # If no threshold, all samples are used.
                filtered_loss = weighted_components # (N, D)
 
            # Apply final reduction (mean, sum, or none)
            if self.reduction == 'mean':
                total_loss = filtered_loss.mean() # Mean over confident samples and D dimensions
            elif self.reduction == 'sum':
                total_loss = filtered_loss.sum() # Sum over confident samples and D dimensions
            else: # 'none'
                total_loss = filtered_loss # Shape: (N_confident, D) or (N, D)
 
        elif task_type == 'classification':
            if uncertainties is not None:
                print("Warning: 'uncertainties' argument is ignored for 'classification' task_type.")
            
            logits = predictions # (N, C)
            
            # Generate pseudo-labels: take the class with the highest logit.
            pseudo_labels = torch.argmax(logits, dim=-1) # (N,)
            
            # Calculate confidence (maximum predicted probability).
            probs = F.softmax(logits, dim=-1) # (N, C)
            max_probs, _ = probs.max(dim=-1) # (N,)
 
            # Compute per-sample cross-entropy loss with pseudo-labels.
            # F.cross_entropy returns average NLL if reduction is not 'none'.
            ce_loss_per_sample = F.cross_entropy(logits, pseudo_labels, reduction='none') # (N,)
 
            # Weights can be simply the maximum probability (higher confidence -> higher weight).
            # Other forms, like (max_probs - min_prob_threshold)^p, could also be used.
            weights_per_sample = max_probs # (N,)
 
            # Weighted loss per sample: (N,)
            weighted_ce_loss_per_sample = ce_loss_per_sample * weights_per_sample
 
            # Filter samples based on confidence threshold
            if self.classification_threshold is not None:
                # Create a mask for samples where max probability exceeds the threshold.
                confident_mask = (max_probs > self.classification_threshold) # (N,)
                
                if confident_mask.sum() == 0:
                    # If no confident samples, return zero loss immediately.
                    return total_loss
                
                # Filter loss for high-confidence samples.
                filtered_loss = weighted_ce_loss_per_sample[confident_mask] # (N_confident,)
            else:
                # If no threshold, all samples are used.
                filtered_loss = weighted_ce_loss_per_sample # (N,)
 
            # Apply final reduction (mean, sum or none)
            if self.reduction == 'mean':
                total_loss = filtered_loss.mean() # Mean over confident samples
            elif self.reduction == 'sum':
                total_loss = filtered_loss.sum() # Sum over confident samples
            else: # 'none'
                total_loss = filtered_loss # Shape: (N_confident,) or (N,)
 
        else:
            raise ValueError(f"Unsupported task_type: '{task_type}'. Must be 'regression' or 'classification'.")
        
        return total_loss
