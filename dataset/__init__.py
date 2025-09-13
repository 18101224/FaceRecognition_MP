from .fer import FER,FER_KFOLD,ClassBatchSampler
from .sampler import ImbalancedDatasetSampler
from .Imbalanced import get_cifar_dataset, Large_dataset
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from .sampler_wrapper import DistributedSamplerWrapper
from .transform import get_transform, random_masking, point_block_mask
from .noise_dataset import get_kfolds, get_loaders, get_noise_dataset
# Public symbols that will be available when using `from dataset import *`
__all__ = ['get_cifar_dataset', 'FER', 'FER_KFOLD', 'get_kfolds', 'get_transform', 'get_loaders', 'DistributedSamplerWrapper'
, 'Large_dataset', 'get_noise_dataset', 'ImbalancedDatasetSampler',
 'ClassBatchSampler', 'random_masking', 'point_block_mask']


def visualize_label_distribution(labels, save_path=None, figsize=(15, 5), class_names=None):
    """
    Visualize the distribution of labels using both bar plot and pie chart.
    
    Args:
        labels (list or np.array): List of integer labels
        save_path (str, optional): If provided, save the figure to this path, should contain extension
        figsize (tuple): Figure size (width, height)
        class_names (list, optional): List of class names corresponding to label indices.
                                     If None, will use label indices as names.
    """
    # Convert labels to numpy array if it's not already
    labels = np.array(labels)
    
    # Count occurrences of each label
    label_counts = Counter(labels)
    
    # Get class names
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(label_counts))]
    else:
        # Ensure class_names matches the number of unique labels
        assert len(class_names) >= len(label_counts), "Number of class names must be >= number of unique labels"
        class_names = [class_names[i] for i in sorted(label_counts.keys())]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot
    sns.barplot(x=list(label_counts.keys()), y=list(label_counts.values()), ax=ax1)
    ax1.set_title('Label Distribution (Bar Plot)')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Number of Samples')
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    
    # Pie chart
    ax2.pie(label_counts.values(), labels=class_names, autopct='%1.1f%%')
    ax2.set_title('Label Distribution (Pie Chart)')
    
    # Calculate and display statistics
    total_samples = len(labels)
    stats_text = (
        f'Total samples: {total_samples}\n'
        f'Number of classes: {len(label_counts)}\n'
        f'Min samples per class: {min(label_counts.values())}\n'
        f'Max samples per class: {max(label_counts.values())}\n'
        f'Mean samples per class: {total_samples/len(label_counts):.1f}'
    )
    
    # Add statistics text to the figure
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()
    
    # Return distribution statistics
    return {
        'class_counts': dict(label_counts),
        'total_samples': total_samples,
        'num_classes': len(label_counts),
        'min_samples': min(label_counts.values()),
        'max_samples': max(label_counts.values()),
        'mean_samples': total_samples / len(label_counts),
        'class_percentages': {k: (v/total_samples)*100 for k, v in label_counts.items()}
    }


