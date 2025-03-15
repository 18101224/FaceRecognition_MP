import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.optimize import minimize
import os

def optimize_sphere_positions(target_angles):
    """
    Optimize positions of points on a sphere to match target angles.
    First point is fixed at (1,0,0).
    
    Args:
        target_angles: torch.Tensor of shape (n_classes, n_classes) containing target angles in radians
    
    Returns:
        numpy array of shape (n_classes, 3) containing optimized 3D positions
    """
    n_classes = target_angles.shape[0]
    
    def angles_to_coords(angles):
        """Convert spherical angles to 3D coordinates"""
        # angles contains pairs of (theta, phi) for each point except first
        points = np.zeros((n_classes, 3))
        points[0] = [1, 0, 0]  # First point fixed at (1,0,0)
        
        for i in range(n_classes - 1):
            theta, phi = angles[2*i:2*i+2]
            points[i+1] = [
                np.cos(phi) * np.cos(theta),
                np.cos(phi) * np.sin(theta),
                np.sin(phi)
            ]
        return points
    
    def loss_function(angles):
        """Compute loss between current angles and target angles"""
        points = angles_to_coords(angles)
        current_cos = np.clip(points @ points.T, -1, 1)
        current_angles = np.arccos(current_cos)
        
        # MSE between current and target angles
        return np.mean((current_angles - target_angles.cpu().numpy())**2)
    
    # Initial guess: distribute points roughly uniformly
    initial_angles = []
    for i in range(1, n_classes):
        theta = 2 * np.pi * i / (n_classes - 1)
        phi = np.pi / 4  # 45 degrees elevation
        initial_angles.extend([theta, phi])
    
    # Optimize positions
    result = minimize(loss_function, initial_angles, method='Powell')
    return angles_to_coords(result.x)

def create_sphere_guides(ax, alpha=0.1):
    """
    Create visual guides for better 3D perception of the sphere.
    
    Args:
        ax: matplotlib 3D axis
        alpha: base transparency for the guides
    """
    # Create more detailed sphere grid
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    phi, theta = np.meshgrid(phi, theta)

    # Sphere surface coordinates
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    # Plot sphere surface with very light color
    ax.plot_surface(x, y, z, color='gray', alpha=0.05)
    
    # Add principal circles (equator and two meridians)
    t = np.linspace(0, 2*np.pi, 100)
    
    # Equator (xy-plane)
    ax.plot(np.cos(t), np.sin(t), np.zeros_like(t), 'gray', alpha=0.3, linestyle='--')
    
    # Principal meridians (xz-plane and yz-plane)
    ax.plot(np.cos(t), np.zeros_like(t), np.sin(t), 'gray', alpha=0.3, linestyle='--')
    ax.plot(np.zeros_like(t), np.cos(t), np.sin(t), 'gray', alpha=0.3, linestyle='--')
    
    # Add coordinate grid lines
    for i in range(-10, 11, 2):
        i = i / 10
        if i != 0:  # Skip the principal circles
            # Parallels
            circle_r = np.sqrt(1 - i**2)
            ax.plot(circle_r * np.cos(t), circle_r * np.sin(t), 
                   np.ones_like(t) * i, 'gray', alpha=0.1)
            
            # Meridians
            x = np.sin(phi) * np.cos(i * np.pi)
            y = np.sin(phi) * np.sin(i * np.pi)
            z = np.cos(phi)
            ax.plot(x, y, z, 'gray', alpha=0.1)

def plot_sphere_distribution(weights, save_path=None, elev=30, azim=45):
    """
    Visualize class weights on a 3D sphere with first class at (1,0,0)
    and other classes positioned to preserve angular relationships.
    
    Args:
        weights: numpy array or torch tensor of shape (n_classes, embed_dim)
        save_path: Optional path to save the plot
        elev: elevation angle for main view
        azim: azimuth angle for main view
    """
    if isinstance(weights, np.ndarray):
        weights = torch.from_numpy(weights)
    weights = torch.nn.functional.normalize(weights, dim=1)
    
    # Compute target angles between all pairs
    cos_sims = torch.mm(weights, weights.T)
    target_angles = torch.acos(torch.clamp(cos_sims, -1.0, 1.0))
    
    # Optimize positions on sphere
    positions = optimize_sphere_positions(target_angles)
    
    # Create figure with multiple views
    fig = plt.figure(figsize=(20, 15))
    
    # Main view (larger)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    create_sphere_guides(ax1)
    
    # Plot points with depth-dependent size and alpha
    distances = positions[:, 0]  # x-coordinate represents depth in main view
    sizes = 100 * (distances + 2) / 3  # larger size for points in front
    alphas = (distances + 2) / 3  # more opaque for points in front
    
    # Plot guide arrows from origin to each point
    for i, pos in enumerate(positions):
        ax1.quiver(0, 0, 0, pos[0], pos[1], pos[2], 
                  color=plt.cm.viridis(i / (len(positions) - 1)),
                  alpha=0.6,
                  arrow_length_ratio=0.1)
    
    # Plot points
    scatter = ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                         color=plt.cm.viridis(np.linspace(0, 1, len(positions))),
                         s=sizes,
                         alpha=alphas)
    
    # Add class labels with depth-dependent properties
    for i, pos in enumerate(positions):
        ax1.text(pos[0]*1.1, pos[1]*1.1, pos[2]*1.1, f' {i}',
                fontsize=12 * (pos[0] + 2) / 3,  # larger font for points in front
                alpha=(pos[0] + 2) / 3,  # more opaque for points in front
                fontweight='bold')
    
    # Add coordinate axes with markers
    for i in range(-10, 11, 2):
        i = i / 10
        ax1.plot([i, i], [0, 0], [-1.2, 1.2], 'gray', alpha=0.1)
        ax1.plot([0, 0], [i, i], [-1.2, 1.2], 'gray', alpha=0.1)
        ax1.plot([-1.2, 1.2], [0, 0], [i, i], 'gray', alpha=0.1)
    
    # Main axes
    ax1.quiver(0, 0, 0, 1.2, 0, 0, color='r', alpha=0.5, label='X')
    ax1.quiver(0, 0, 0, 0, 1.2, 0, color='g', alpha=0.5, label='Y')
    ax1.quiver(0, 0, 0, 0, 0, 1.2, color='b', alpha=0.5, label='Z')
    
    ax1.set_title('Main View', pad=20)
    ax1.view_init(elev=elev, azim=azim)
    
    # Additional views from different angles
    views = [
        (0, 0, 2, 'Front View (YZ Plane)'),
        (0, 90, 3, 'Side View (XZ Plane)'),
        (90, 90, 4, 'Top View (XY Plane)')
    ]
    
    for elev_, azim_, pos, title in views:
        ax = fig.add_subplot(2, 2, pos, projection='3d')
        create_sphere_guides(ax)
        
        # Plot guide arrows in additional views
        for i, pos_i in enumerate(positions):
            ax.quiver(0, 0, 0, pos_i[0], pos_i[1], pos_i[2], 
                     color=plt.cm.viridis(i / (len(positions) - 1)),
                     alpha=0.6,
                     arrow_length_ratio=0.1)
        
        # Plot points
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  color=plt.cm.viridis(np.linspace(0, 1, len(positions))),
                  s=50)
        
        # Add minimal labels
        for i, pos_i in enumerate(positions):
            ax.text(pos_i[0]*1.1, pos_i[1]*1.1, pos_i[2]*1.1, f' {i}',
                   fontsize=8,
                   fontweight='bold')
        
        ax.quiver(0, 0, 0, 1.2, 0, 0, color='r', alpha=0.5)
        ax.quiver(0, 0, 0, 0, 1.2, 0, color='g', alpha=0.5)
        ax.quiver(0, 0, 0, 0, 0, 1.2, color='b', alpha=0.5)
        
        ax.view_init(elev=elev_, azim=azim_)
        ax.set_title(title)
    
    # Common settings for all subplots
    for ax in fig.get_axes():
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])
        ax.set_box_aspect([1,1,1])
    
    # Adjust layout
    plt.suptitle('Class Centers on Unit Sphere\nFirst Class at (1,0,0)', y=0.95, fontsize=16)
    plt.tight_layout()
    
    # Print angular accuracy statistics
    realized_cos = torch.from_numpy(positions @ positions.T)
    realized_angles = torch.acos(torch.clamp(realized_cos, -1.0, 1.0))
    angle_errors = (realized_angles - target_angles).abs() * 180 / np.pi
    
    print("\nAngle preservation statistics:")
    print(f"Mean absolute error: {angle_errors.mean():.2f}°")
    print(f"Max absolute error: {angle_errors.max():.2f}°")
    print(f"Std of absolute error: {angle_errors.std():.2f}°")
    
    # Save or show plot
    if save_path:
        os.makedirs('results', exist_ok=True)
        plt.savefig(f'results/{save_path}', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
    return positions

def plot_angle_preservation(weights, positions):
    """
    Plot how well the sphere visualization preserves the original angles.
    
    Args:
        weights: original weight vectors
        positions: optimized 3D positions
    """
    # Compute original and realized angles
    if isinstance(weights, np.ndarray):
        weights = torch.from_numpy(weights)
    weights = torch.nn.functional.normalize(weights, dim=1)
    
    original_cos = torch.mm(weights, weights.T)
    original_angles = torch.acos(torch.clamp(original_cos, -1.0, 1.0)) * 180 / np.pi
    
    realized_cos = torch.from_numpy(positions @ positions.T)
    realized_angles = torch.acos(torch.clamp(realized_cos, -1.0, 1.0)) * 180 / np.pi
    
    # Create scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(original_angles.flatten().cpu(), 
               realized_angles.flatten().cpu(),
               alpha=0.5)
    
    # Add diagonal line
    max_angle = max(original_angles.max(), realized_angles.max())
    plt.plot([0, max_angle], [0, max_angle], 'r--', label='Perfect Preservation')
    
    plt.xlabel('Original Angles (degrees)')
    plt.ylabel('Realized Angles (degrees)')
    plt.title('Angle Preservation in Sphere Visualization')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_angle_matrix(angles_rad, save_path=None):
    """
    Plot the matrix of angles between class centers as a heatmap.
    
    Args:
        angles_rad: torch.Tensor of shape (n_classes, n_classes) containing angles in radians
        save_path: Optional path to save the plot
    """
    # Convert angles to degrees
    angles_deg = angles_rad.cpu().numpy() * 180 / np.pi
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(angles_deg, 
                annot=True, 
                fmt='.1f', 
                cmap='viridis',
                xticklabels=range(len(angles_deg)),
                yticklabels=range(len(angles_deg)),
                cbar_kws={'label': 'Angle (degrees)'})
    
    # Add labels and title
    plt.xlabel('Class Index')
    plt.ylabel('Class Index')
    plt.title('Angular Distances Between Class Centers')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show plot
    if save_path:
        os.makedirs('results', exist_ok=True)
        plt.savefig(f'results/{save_path}', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def analyze_sphere_distribution(weights, sphere_save_path=None, matrix_save_path=None, elev=30, azim=45):
    """
    Analyze and visualize the distribution of class centers on a sphere and their angular relationships.
    
    Args:
        weights: numpy array or torch tensor of shape (n_classes, embed_dim)
        sphere_save_path: Optional path to save the sphere visualization
        matrix_save_path: Optional path to save the angle matrix heatmap
        elev: elevation angle for main sphere view
        azim: azimuth angle for main sphere view
    """
    if isinstance(weights, np.ndarray):
        weights = torch.from_numpy(weights)
    weights = torch.nn.functional.normalize(weights, dim=1)
    
    # Compute target angles between all pairs
    cos_sims = torch.mm(weights, weights.T)
    target_angles = torch.acos(torch.clamp(cos_sims, -1.0, 1.0))
    
    # Plot sphere visualization
    positions = plot_sphere_distribution(weights, save_path=sphere_save_path, elev=elev, azim=azim)
    
    # Plot angle matrix
    plot_angle_matrix(target_angles, save_path=matrix_save_path)
    
    # Print detailed statistics
    angles_deg = target_angles.cpu().numpy() * 180 / np.pi
    
    print("\nDetailed angular statistics:")
    print("\nMean angles from each class to others:")
    for i in range(len(angles_deg)):
        other_angles = angles_deg[i][angles_deg[i] > 0]  # Exclude self-angle (0)
        print(f"Class {i}:")
        print(f"  Mean: {other_angles.mean():.2f}°")
        print(f"  Min:  {other_angles.min():.2f}°")
        print(f"  Max:  {other_angles.max():.2f}°")
        print(f"  Std:  {other_angles.std():.2f}°")
    
    return positions, target_angles  