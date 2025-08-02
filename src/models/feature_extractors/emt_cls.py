# src/models/feature_extractors/emt_cls.py
# CBCR License 1.0
#
# Copyright 2022 Centre for Brain Computing Research (CBCR)
#
# Redistribution and use for non-commercial purpose in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials 
# provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior 
# written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF 
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTER-
# RUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# This code is an adaptation of the EmT model, as described in:
# "EmT: Emotion Transformer for Cross-subject Emotion Recognition from EEG"
# https://arxiv.org/abs/2406.18345

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.utils import weight_norm
from einops import rearrange
import math

# --- Graph Convolutional Layers ---

class GCN(Module):
    """
    A simple Graph Convolutional Network layer, as described in https://arxiv.org/abs/1609.02907.
    It expects a tuple of (features, adjacency_matrix) as input.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights and biases."""
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, data: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            data (tuple): A tuple containing:
                - graph (torch.Tensor): Node features of shape (batch, num_nodes, in_features).
                - adj (torch.Tensor): Adjacency matrix of shape (num_nodes, num_nodes).
        Returns:
            tuple: A tuple containing the updated features and the original adjacency matrix.
        """
        graph, adj = data
        adj_norm = self._normalize_adj(adj)
        support = torch.matmul(graph, self.weight)
        output = torch.matmul(adj_norm, support)
        
        if self.bias is not None:
            output = F.relu(output + self.bias)
        else:
            output = F.relu(output)
            
        return output, adj

    def _normalize_adj(self, adj: torch.Tensor) -> torch.Tensor:
        """Symmetrically normalize adjacency matrix."""
        rowsum = torch.sum(adj, dim=-1)
        # Add a small epsilon for nodes with no connections to avoid division by zero
        mask = torch.zeros_like(rowsum)
        mask[rowsum == 0] = 1
        rowsum += mask
        
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        
        # Ensure adj is broadcastable if it has a batch dimension
        if adj.dim() == 3: # (batch, N, N)
            return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        else: # (N, N)
            return torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)


# --- Attention and Transformer Blocks ---

class Attention(nn.Module):
    """Spatial-Temporal Attention module."""
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, anchor: int = 3, dropout: float = 0., alpha: float = 0.25):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # Temporal convolution to capture local sequence patterns
        self.STA = nn.Sequential(
            nn.Dropout(alpha * dropout),
            weight_norm(nn.Conv2d(self.heads, self.heads, (anchor, 1),
                                  stride=1, padding=(int(0.5 * (anchor - 1)), 0))),
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = self.STA(out)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    """Standard Feed-Forward Network."""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PreNorm(nn.Module):
    """Layer normalization followed by a function."""
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class TemporalTransformer(nn.Module):
    """Transformer block for temporal feature aggregation."""
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0., alpha: float = 0.25):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, alpha=alpha)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# --- Graph Encoder Modules ---

class GraphEncoder(nn.Module):
    """Encoder that applies a series of GCN layers."""
    def __init__(self, num_layers: int, num_node: int, in_features: int, out_features: int):
        super(GraphEncoder, self).__init__()
        layers = []
        for i in range(num_layers):
            in_dim = in_features if i == 0 else out_features
            layers.append(GCN(in_dim, out_features))
        self.encoder = nn.Sequential(*layers)
        self.tokenizer = nn.Linear(num_node * out_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, num_nodes, in_features)
        # adj: (num_nodes, num_nodes)
        output, _ = self.encoder((x, adj))
        
        # Tokenize graph features into a single vector
        output = output.reshape(output.size(0), -1)
        output = self.tokenizer(output)
        return output


# --- Main EMT Feature Extractor Class ---

class EMTFeatureExtractor(nn.Module):
    def __init__(self, 
                 in_channels: int = 62, 
                 num_features: int = 5,
                 embedding_dim: int = 256,
                 layers_graph: list[int] = [1, 2],
                 layers_transformer: int = 1,
                 num_adj: int = 3,
                 hidden_graph: int = 16,
                 num_head: int = 8,
                 dim_head: int = 16,
                 dropout: float = 0.25,
                 alpha: float = 0.25):
        """
        EMT: Emotion Transformer Feature Extractor.
        This model extracts features from multi-channel EEG signals with pre-computed features.

        Args:
            in_channels (int): Number of EEG channels (nodes in the graph). Default is 62.
            num_features (int): Number of features per channel at each time step (e.g., 5 for DE bands). Default is 5.
            embedding_dim (int): The dimension of the final output feature vector. Default is 256.
            layers_graph (list[int]): Number of GCN layers in each of the two graph encoders. Default is [1, 2].
            layers_transformer (int): Number of layers in the temporal transformer. Default is 1.
            num_adj (int): Number of learnable adjacency matrices (views). Default is 3.
            hidden_graph (int): The hidden dimension of the graph encoder outputs. Default is 16.
            num_head (int): Number of attention heads in the transformer. Default is 8.
            dim_head (int): Dimension of each attention head. Default is 16.
            dropout (float): Dropout rate. Default is 0.25.
            alpha (float): Dropout multiplier for the STA module. Default is 0.25.
        """
        super(EMTFeatureExtractor, self).__init__()
        self.output_dim = embedding_dim
        
        # Multi-view Graph Encoders
        self.ge1 = GraphEncoder(
            num_layers=layers_graph[0], num_node=in_channels, 
            in_features=num_features, out_features=hidden_graph
        )
        self.ge2 = GraphEncoder(
            num_layers=layers_graph[1], num_node=in_channels, 
            in_features=num_features, out_features=hidden_graph
        )

        # Learnable adjacency matrices for different graph views
        self.adjs = Parameter(torch.FloatTensor(num_adj, in_channels, in_channels))
        nn.init.xavier_uniform_(self.adjs)

        # Residual connection path for GNN block
        self.to_gnn_out = nn.Linear(in_channels * num_features, hidden_graph, bias=False)

        # Temporal context aggregation
        self.transformer = TemporalTransformer(
            depth=layers_transformer, dim=hidden_graph, heads=num_head,
            dim_head=dim_head, dropout=dropout, mlp_dim=dim_head, alpha=alpha
        )

        # Final fully connected layer to map to the desired embedding dimension
        self.fc_out = nn.Linear(hidden_graph, self.output_dim)

    def _get_adj(self, self_loop: bool = True) -> torch.Tensor:
        """
        Generates symmetric adjacency matrices from the learnable parameters.
        
        Args:
            self_loop (bool): If True, adds an identity matrix to include self-connections.
        
        Returns:
            torch.Tensor: The processed adjacency matrices.
        """
        # Symmetrize by adding the transpose
        adj = F.relu(self.adjs + self.adjs.transpose(2, 1))
        if self_loop:
            # Add self-connections using an identity matrix on the correct device
            identity = torch.eye(adj.size(1), device=adj.device, dtype=adj.dtype)
            adj = adj + identity.unsqueeze(0) # Add batch dim for broadcasting
        return adj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the EMT Feature Extractor.

        Args:
            x (torch.Tensor): Input tensor with shape (batch, channels, sequence, features).
                               Example: (64, 62, 36, 5)
        Returns:
            torch.Tensor: Output feature embedding of shape (batch, embedding_dim).
        """
        # Permute to (batch, sequence, channels, features) to match original model's expectation
        x = x.permute(0, 2, 1, 3) 
        b, s, c, f = x.size()

        # Reshape for processing each time step through the graph encoders
        x_reshaped = rearrange(x, 'b s c f -> (b s) c f')
        
        adjs = self._get_adj()

        # Multi-view pyramid residual GNN block
        # 1. Residual path
        x_residual = x_reshaped.reshape(b * s, -1)
        x_residual = self.to_gnn_out(x_residual)
        
        # 2. Graph encoder paths (two views)
        x1 = self.ge1(x_reshaped, adjs[0])
        x2 = self.ge2(x_reshaped, adjs[1])
        
        # 3. Combine the three paths
        x_graph_features = torch.stack((x_residual, x1, x2), dim=1)
        x_graph_features = torch.mean(x_graph_features, dim=1) # (b*s, hidden_graph)
        
        # Reshape for the temporal transformer
        x_temporal = rearrange(x_graph_features, '(b s) h -> b s h', b=b, s=s)
        
        # Apply temporal transformer and aggregate sequence features
        x_temporal = self.transformer(x_temporal)
        x_aggregated = torch.mean(x_temporal, dim=1) # Average pooling over sequence dimension

        # Final mapping to the embedding dimension
        output = self.fc_out(x_aggregated)
        
        return output

# --- Factory Function ---

def build_emt_extractor(
    in_channels: int = 62,
    num_features: int = 5,
    embedding_dim: int = 256,
    **kwargs
) -> EMTFeatureExtractor:
    """
    Builds an EMTFeatureExtractor model with specified configurations.

    Args:
        in_channels (int): Number of EEG channels.
        num_features (int): Number of features per channel-timestep.
        embedding_dim (int): Dimension of the output feature vector.
        **kwargs: Additional keyword arguments for the EMTFeatureExtractor constructor.

    Returns:
        EMTFeatureExtractor: The instantiated model.
    """
    return EMTFeatureExtractor(
        in_channels=in_channels,
        num_features=num_features,
        embedding_dim=embedding_dim,
        **kwargs
    )

# --- Example Usage and Verification ---

if __name__ == "__main__":
    def count_parameters(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    BATCH_SIZE = 16
    CHANNELS = 62
    SEQUENCE_LENGTH = 36
    NUM_FEATURES = 5
    EMBEDDING_DIM = 256

    # Shape: (batch, channels, sequence, features)
    dummy_data = torch.randn(BATCH_SIZE, CHANNELS, SEQUENCE_LENGTH, NUM_FEATURES)

    emt_extractor = build_emt_extractor(
        in_channels=CHANNELS,
        num_features=NUM_FEATURES,
        embedding_dim=EMBEDDING_DIM,
        hidden_graph=32,
        layers_transformer=2,
        num_head=8,
        dim_head=16
    )
    
    print("EMT Feature Extractor Architecture:")
    print(emt_extractor)
    print(f"\nTotal trainable parameters: {count_parameters(emt_extractor):,}")

    # Forward pass
    with torch.no_grad():
        output_features = emt_extractor(dummy_data)

    print(f"\nInput shape:  {dummy_data.shape}")
    print(f"Output shape: {output_features.shape}")
    assert output_features.shape == (BATCH_SIZE, EMBEDDING_DIM), "Output shape is incorrect!"
    print("\nModel instantiation and forward pass successful!")