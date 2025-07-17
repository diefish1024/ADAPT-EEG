# src/utils/data_utils.py

import torch
import torch.nn.functional as F

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length EEG features.
    It pads the features to the maximum length within the current batch.
    Assumes batch is a list of (features_tensor, label_tensor) tuples.
    Features_tensor is expected to be [channels, N_samples_or_features].
    """
    features_list = [item[0] for item in batch] # List of [C, T_i]
    labels_list = [item[1] for item in batch]   # List of labels

    # Determine the maximum sequence length in the current batch
    max_seq_len = max(f.shape[1] for f in features_list)

    # Pad features to max_seq_len
    padded_features = []
    for features_tensor in features_list:
        current_seq_len = features_tensor.shape[1]
        if current_seq_len < max_seq_len:
            # Calculate padding amount for the last dimension
            padding = [0, max_seq_len - current_seq_len] # (padding_left, padding_right) for last dim
            # Pad with zeros
            padded_tensor = F.pad(features_tensor, padding)
            padded_features.append(padded_tensor)
        else:
            padded_features.append(features_tensor)

    # Stack the padded features and labels to form batch tensors
    stacked_features = torch.stack(padded_features, 0) # Stacks along a new batch dimension
    stacked_labels = torch.stack(labels_list, 0)     # Labels usually have consistent shape

    return stacked_features, stacked_labels
