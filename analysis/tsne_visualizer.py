import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_weight_tsne(weights, save_path=None):
    """
    Visualize weight matrix distribution using t-SNE.
    
    Args:
        weights: numpy array or torch tensor of shape (n_classes, embed_dim)
        save_path: Optional path to save the plot
    """
    # Convert weights to numpy if it's a torch tensor
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, weights.shape[0]-1))
    weights_tsne = tsne.fit_transform(weights)
    
    # Create the visualization
    plt.figure(figsize=(10, 10))
    
    # Create scatter plot
    scatter = plt.scatter(weights_tsne[:, 0], weights_tsne[:, 1], 
                         c=np.arange(weights.shape[0]), 
                         cmap='viridis', 
                         s=100)
    
    # Add class labels with index
    for i in range(weights.shape[0]):
        plt.annotate(f'{i}', # Changed to just show index number
                    (weights_tsne[i, 0], weights_tsne[i, 1]),
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=12,
                    fontweight='bold')
    
    # Add colorbar
    plt.colorbar(scatter, label='Class Index')
    
    # Add title and labels
    plt.title('t-SNE Visualization of Weight Matrix')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    
    # Add circle
    circle = plt.Circle((0, 0), 
                       np.mean(np.sqrt(np.sum(weights_tsne**2, axis=1))), 
                       fill=False, 
                       color='red', 
                       linestyle='--', 
                       label='Mean Distance Circle')
    plt.gca().add_artist(circle)
    plt.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def analyze_weight_matrix(weights):
    """
    Analyze weight matrix and print statistics
    
    Args:
        weights: numpy array or torch tensor of shape (n_classes, embed_dim)
    """
    # Convert to torch tensor if numpy array
    if isinstance(weights, np.ndarray):
        weights = torch.from_numpy(weights)
    
    # Calculate statistics
    norm = torch.norm(weights, dim=1)
    weights_normalized = torch.nn.functional.normalize(weights, dim=1)
    cosine_sim = torch.mm(weights_normalized, weights_normalized.T)
    
    print(f"Weight matrix shape: {weights.shape}")
    print(f"Number of classes: {weights.shape[0]}")
    print(f"Embedding dimension: {weights.shape[1]}")
    
    print("\nWeight norms:")
    print(f"Mean: {norm.mean().item():.4f}")
    print(f"Std: {norm.std().item():.4f}")
    print(f"Min: {norm.min().item():.4f}")
    print(f"Max: {norm.max().item():.4f}")
    
    # Analyze inter-class relationships
    cosine_sim.fill_diagonal_(float('-inf'))
    max_sim, _ = torch.max(cosine_sim, dim=1)
    
    print("\nInter-class cosine similarities:")
    print(f"Mean: {max_sim.mean().item():.4f}")
    print(f"Std: {max_sim.std().item():.4f}")
    print(f"Min: {max_sim.min().item():.4f}")
    print(f"Max: {max_sim.max().item():.4f}")

