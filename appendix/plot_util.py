import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torch import Tensor
import torch
from torch import Tensor


__all__ = [
    "plot_micro_accuracy_histogram",
    "plot_accuracy_histogram",
    "plot_confusion_matrix",
    "plot_angle_with_confusion"
]


def plot_micro_accuracy_histogram(micro_acc1, micro_acc2, labels=("Model 1", "Model 2"), title="Micro Accuracy Comparison", save_path=None):
    """
    Plot a bar chart comparing the micro accuracy of two models.

    Args:
        micro_acc1 (float): Micro accuracy of the first model.
        micro_acc2 (float): Micro accuracy of the second model.
        labels (tuple): Labels for the two models.
        title (str): Title of the plot.
        save_path (str, optional): If provided, save the figure to this path.
    """
    plt.figure(figsize=(6, 6))
    accs = [micro_acc1, micro_acc2]
    x = np.arange(len(accs))
    barlist = plt.bar(x, accs, color=['skyblue', 'lightcoral'], edgecolor='black')
    plt.xticks(x, labels)
    plt.ylim(0, 1.1)
    plt.ylabel("Micro Accuracy")
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels on top of bars
    for i, v in enumerate(accs):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_accuracy_histogram(accuracies1,save_path, accuracies2=None, labels=None, title="Class-wise Accuracy Comparison"):
    """
    Plot histogram(s) of class-wise accuracies.
    
    Args:
        accuracies1 (list): List of accuracies for the first model
        accuracies2 (list, optional): List of accuracies for the second model for comparison
        labels (tuple, optional): Labels for the two models (e.g., ("Model A", "Model B"))
        title (str): Title of the plot
    """
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(accuracies1))
    width = 0.35 if accuracies2 is not None else 0.6
    
    if accuracies2 is None:
        # Single histogram
        plt.bar(x, accuracies1, width, color='skyblue', edgecolor='black')
        plt.xlabel('Class Index')
        plt.ylabel('Accuracy')
        plt.title(title)
    else:
        # Comparative histogram
        if labels is None:
            labels = ("Model 1", "Model 2")
            
        plt.bar(x - width/2, accuracies1, width, label=labels[0], color='skyblue', edgecolor='black')
        plt.bar(x + width/2, accuracies2, width, label=labels[1], color='lightcoral', edgecolor='black')
        
        plt.xlabel('Class Index')
        plt.ylabel('Accuracy')
        plt.title(title)
        plt.legend()
    
    plt.xticks(x)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.1)  # Set y-axis limit from 0 to 1.1 to show full range of accuracies
    
    # Add value labels on top of bars
    def add_value_labels(accuracies, x_pos):
        for i, v in enumerate(accuracies):
            plt.text(x_pos[i], v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    
    if accuracies2 is None:
        add_value_labels(accuracies1, x)
    else:
        add_value_labels(accuracies1, x - width/2)
        add_value_labels(accuracies2, x + width/2)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(confusion_matrix, save_path, class_names=None, title="Confusion Matrix", 
                         cmap="Blues", figsize=(10, 8), normalize=False):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        confusion_matrix (numpy.ndarray): 2D array of confusion matrix
        class_names (list, optional): List of class names. If None, will use indices
        title (str): Title of the plot
        cmap (str): Colormap for the heatmap
        figsize (tuple): Figure size (width, height)
        normalize (bool): If True, normalize the confusion matrix
    """
    if normalize:
        # Normalize by row (true labels)
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    
    # Create heatmap
    ax = sns.heatmap(confusion_matrix, annot=True, fmt=fmt, cmap=cmap,
                     xticklabels=class_names if class_names else 'auto',
                     yticklabels=class_names if class_names else 'auto')
    
    # Set labels and title
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_angle_with_confusion(angle_matrix: Tensor, preds: Tensor, labels: Tensor, save_path: str):
    """
    Plot angle heatmap and confusion matrix side by side.
    
    Args:
        angle_matrix (Tensor): (#_classes, #_classes) shaped tensor containing angles
        preds (Tensor): Predictions tensor
        labels (Tensor): Ground truth labels tensor
        save_path (str): Path to save the figure
    """
    # Calculate angle statistics
    i, j = torch.triu_indices(angle_matrix.shape[0], angle_matrix.shape[0], offset=1)
    up = angle_matrix[i,j].reshape(-1)
    angle_mean, angle_std = up.mean().item(), up.std().item()
    
    # Convert tensors to numpy arrays
    angle_matrix = angle_matrix
    preds = torch.max(preds, dim=1)[1]
    preds = preds.numpy()
    labels = labels.numpy()
    
    # Create confusion matrix
    num_classes = angle_matrix.shape[0]
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    
    for p, l in zip(preds, labels):
        confusion_matrix[l, p] += 1
    
    # Create figure with two subplots side by side
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.1, 1])  # Add small space for text
    
    ax1 = fig.add_subplot(gs[0])
    ax_text = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    
    # Plot angle heatmap
    sns.heatmap(angle_matrix, 
                ax=ax1,
                cmap='coolwarm',
                vmin=80.0,
                vmax=120.0,
                annot=True,
                fmt='.1f',
                cbar_kws={'label': 'Angle (degrees)'})
    ax1.set_title('Angle Matrix')
    ax1.set_xlabel('Class Index')
    ax1.set_ylabel('Class Index')
    
    # Add statistics text
    ax_text.axis('off')  # Hide axes
    stats_text = f'Angle Statistics:\nMean: {angle_mean:.2f}°\nStd: {angle_std:.2f}°'
    ax_text.text(0.5, 0.5, stats_text,
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax_text.transAxes,
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Calculate normalized confusion matrix (error ratio per class)
    row_sums = confusion_matrix.sum(axis=1)  # Remove keepdims=True
    normalized_confusion = np.zeros_like(confusion_matrix, dtype=np.float32)
    # Avoid division by zero
    for i in range(num_classes):
        if row_sums[i] > 0:
            normalized_confusion[i, :] = confusion_matrix[i, :] / row_sums[i]
    
    # Plot normalized confusion matrix
    sns.heatmap(normalized_confusion,
                ax=ax2,
                cmap='YlOrRd',  # Yellow to Orange to Red colormap
                vmin=0,
                vmax=1,
                annot=True,
                fmt='.2f',  # Show 2 decimal places
                cbar_kws={'label': 'Error Ratio'})
    ax2.set_title('Normalized Confusion Matrix\n(Error Ratio per Class)')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    
    # Add class sample counts to y-axis labels
    y_labels = [f'Class {i}\n(n={row_sums[i]})' for i in range(num_classes)]
    ax2.set_yticklabels(y_labels)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    
    