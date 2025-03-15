import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_angular_distribution(weights, save_path=None):
    """
    Visualize class weights based on their angular relationships with the first class.
    Places the first class at (1,0) and other classes based on their angles.
    
    Args:
        weights: numpy array or torch tensor of shape (n_classes, embed_dim)
        save_path: Optional path to save the plot
    """
    # Convert weights to normalized torch tensor if needed
    if isinstance(weights, np.ndarray):
        weights = torch.from_numpy(weights)
    weights = torch.nn.functional.normalize(weights, dim=1)
    
    # Get cosine similarities with first class
    cos_sims = torch.mm(weights, weights[0:1].T).squeeze()
    # Get angles in radians
    angles = torch.acos(torch.clamp(cos_sims, -1.0, 1.0))
    
    # Project points onto 2D circle
    # First class will be at (1,0)
    x = torch.cos(angles)
    y = torch.sin(angles)
    
    # Create the visualization
    plt.figure(figsize=(10, 10))
    
    # Create scatter plot
    scatter = plt.scatter(x.cpu(), y.cpu(), 
                         c=np.arange(weights.shape[0]), 
                         cmap='viridis', 
                         s=100)
    
    # Add class labels
    for i in range(weights.shape[0]):
        plt.annotate(f'{i}',
                    (x[i].item(), y[i].item()),
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=12,
                    fontweight='bold')
    
    # Add colorbar
    plt.colorbar(scatter, label='Class Index')
    
    # Add title and labels
    plt.title('Angular Distribution of Class Centers\nRelative to First Class')
    plt.xlabel('cos(θ)')
    plt.ylabel('sin(θ)')
    
    # Add unit circle
    circle = plt.Circle((0, 0), 1, 
                       fill=False, 
                       color='red', 
                       linestyle='--', 
                       label='Unit Circle')
    plt.gca().add_artist(circle)
    
    # Add first class reference line
    plt.axhline(y=0, color='gray', linestyle=':')
    plt.axvline(x=0, color='gray', linestyle=':')
    
    # Set equal aspect ratio and limits
    plt.axis('equal')
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    
    # Add legend
    plt.legend()
    
    # Print angular statistics
    print("\nAngular statistics (relative to first class):")
    angles_deg = angles.cpu().numpy() * 180 / np.pi
    print(f"Mean angle: {angles_deg[1:].mean():.2f}°")
    print(f"Min angle: {angles_deg[1:].min():.2f}°")
    print(f"Max angle: {angles_deg[1:].max():.2f}°")
    print(f"Std angle: {angles_deg[1:].std():.2f}°")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(f'results/{save_path}')
        plt.close()
    else:
        plt.show()
        
    return angles_deg

def analyze_angular_distribution(weights):
    """
    Analyze the angular distribution between all pairs of class centers.
    
    Args:
        weights: numpy array or torch tensor of shape (n_classes, embed_dim)
    """
    # Convert to normalized torch tensor if needed
    if isinstance(weights, np.ndarray):
        weights = torch.from_numpy(weights)
    weights = torch.nn.functional.normalize(weights, dim=1)
    
    # Compute all pairwise cosine similarities
    cos_sims = torch.mm(weights, weights.T)
    # Get angles in degrees
    angles = torch.acos(torch.clamp(cos_sims, -1.0, 1.0)) * 180 / np.pi
    
    # Create angle statistics for each class
    print("\nPer-class angular statistics:")
    for i in range(weights.shape[0]):
        class_angles = angles[i]
        class_angles = class_angles[class_angles > 0]  # Remove self-similarity
        print(f"\nClass {i}:")
        print(f"Mean angle to other classes: {class_angles.mean():.2f}°")
        print(f"Min angle to other classes: {class_angles.min():.2f}°")
        print(f"Max angle to other classes: {class_angles.max():.2f}°")
        print(f"Std angle to other classes: {class_angles.std():.2f}°")
    
    # Overall statistics
    angles_no_diag = angles[angles > 0]  # Remove self-similarities
    print("\nOverall angular statistics:")
    print(f"Mean angle between all pairs: {angles_no_diag.mean():.2f}°")
    print(f"Min angle between all pairs: {angles_no_diag.min():.2f}°")
    print(f"Max angle between all pairs: {angles_no_diag.max():.2f}°")
    print(f"Std angle between all pairs: {angles_no_diag.std():.2f}°") 