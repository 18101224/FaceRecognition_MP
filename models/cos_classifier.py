import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class CosClassifier(nn.Module):
    """
    Cosine Similarity based Classifier with multiple linear layers
    This classifier computes cosine similarity between input features and class centers (weights)
    using iteration through multiple linear layers
    """
    def __init__(self, in_features, num_classes, num_layers=10):
        """
        Args:
            in_features (int): Dimension of input features
            num_classes (int): Number of classes
            num_layers (int): Number of linear layers to use
        """
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_layers = num_layers
        
        # Create ModuleList of linear layers
        self.kernels = nn.ModuleList([
            nn.Linear(in_features, num_classes, bias=False)
            for _ in range(num_layers)
        ])
        
        # Initialize all layers
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights using Kaiming uniform initialization"""
        for kernel in self.kernels:
            nn.init.kaiming_uniform_(kernel.weight, a=np.sqrt(5))
    
    def forward(self, features):
        """
        Compute cosine similarities between input features and class centers
        using iteration through multiple linear layers
        
        Args:
            features (torch.Tensor): Input features of shape (batch_size, in_features)
            
        Returns:
            torch.Tensor: Average cosine similarities of shape (batch_size, num_classes)
        """
        # Normalize feature vectors
        features_norm = F.normalize(features, p=2, dim=1)
        
        # Initialize accumulator for cosine similarities
        cos_sims = []
        
        # Iterate through all kernels
        for kernel in self.kernels:
            # Get the weights and normalize them
            kernel_weights = F.normalize(kernel.weight, p=2, dim=1)  # (num_classes, in_features)
            
            # Compute cosine similarity for this layer
            # (batch_size, in_features) @ (in_features, num_classes) -> (batch_size, num_classes)
            cos_sim = features_norm @ kernel_weights.t()
            cos_sims.append(cos_sim)
        
        # Average the cosine similarities from all layers
        avg_cos_sim = torch.stack(cos_sims, dim=0).mean(dim=0)
        return avg_cos_sim
    
    def extra_repr(self):
        """String representation of the module"""
        return f'in_features={self.in_features}, num_classes={self.num_classes}, num_layers={self.num_layers}' 