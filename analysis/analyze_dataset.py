import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
from PIL import Image
import torch
from torchvision import transforms
from collections import Counter
import pandas as pd

def analyze_dataset_structure(root_dir):
    """Analyze the structure of the dataset."""
    train_dir = os.path.join(root_dir, 'train')
    valid_dir = os.path.join(root_dir, 'valid')
    
    # Get class directories
    train_classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    valid_classes = sorted([d for d in os.listdir(valid_dir) if os.path.isdir(os.path.join(valid_dir, d))])
    
    # Count images per class
    train_counts = {}
    valid_counts = {}
    
    for cls in train_classes:
        train_counts[cls] = len(os.listdir(os.path.join(train_dir, cls)))
    
    for cls in valid_classes:
        valid_counts[cls] = len(os.listdir(os.path.join(valid_dir, cls)))
    
    return train_counts, valid_counts

def analyze_image_statistics(root_dir):
    """Analyze basic statistics of images in the dataset."""
    train_dir = os.path.join(root_dir, 'train')
    valid_dir = os.path.join(root_dir, 'valid')
    
    # Image statistics
    image_sizes = []
    image_channels = []
    
    # Process training images
    for cls in os.listdir(train_dir):
        cls_dir = os.path.join(train_dir, cls)
        if os.path.isdir(cls_dir):
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                try:
                    img = Image.open(img_path)
                    image_sizes.append(img.size)
                    image_channels.append(len(img.getbands()))
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    return image_sizes, image_channels

def analyze_class_distribution(root_dir, output_filename='class_distribution.png'):
    """Analyze and plot the distribution of classes in train and validation sets."""
    train_dir = os.path.join(root_dir, 'train')
    valid_dir = os.path.join(root_dir, 'valid')
    
    # Get class directories
    train_classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    valid_classes = sorted([d for d in os.listdir(valid_dir) if os.path.isdir(os.path.join(valid_dir, d))])
    
    # Count images per class
    train_counts = {}
    valid_counts = {}
    
    for cls in train_classes:
        train_counts[cls] = len(os.listdir(os.path.join(train_dir, cls)))
    
    for cls in valid_classes:
        valid_counts[cls] = len(os.listdir(os.path.join(valid_dir, cls)))
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Class': list(train_counts.keys()),
        'Train Count': list(train_counts.values()),
        'Valid Count': list(valid_counts.values())
    })
    
    # Calculate imbalance ratios
    total_train = sum(train_counts.values())
    total_valid = sum(valid_counts.values())
    
    df['Train Percentage'] = df['Train Count'] / total_train * 100
    df['Valid Percentage'] = df['Valid Count'] / total_valid * 100
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    
    # Plot absolute counts
    plt.subplot(1, 2, 1)
    x = np.arange(len(df))
    width = 0.35
    
    plt.bar(x - width/2, df['Train Count'], width, label='Train', color='skyblue')
    plt.bar(x + width/2, df['Valid Count'], width, label='Valid', color='lightcoral')
    
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution (Absolute Counts)')
    plt.xticks(x, df['Class'])
    plt.legend()
    
    # Plot percentages
    plt.subplot(1, 2, 2)
    plt.bar(x - width/2, df['Train Percentage'], width, label='Train', color='skyblue')
    plt.bar(x + width/2, df['Valid Percentage'], width, label='Valid', color='lightcoral')
    
    plt.xlabel('Class')
    plt.ylabel('Percentage of Images')
    plt.title('Class Distribution (Percentages)')
    plt.xticks(x, df['Class'])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print imbalance statistics
    print("\nClass Distribution Statistics:")
    print("\nAbsolute Counts:")
    print(df[['Class', 'Train Count', 'Valid Count']].to_string(index=False))
    print("\nPercentages:")
    print(df[['Class', 'Train Percentage', 'Valid Percentage']].round(2).to_string(index=False))
    
    # Calculate and print imbalance metrics
    train_imbalance = df['Train Count'].max() / df['Train Count'].min()
    valid_imbalance = df['Valid Count'].max() / df['Valid Count'].min()
    
    print(f"\nImbalance Ratios:")
    print(f"Training set: {train_imbalance:.2f}x (max/min)")
    print(f"Validation set: {valid_imbalance:.2f}x (max/min)")

def plot_image_statistics(image_sizes, image_channels):
    """Plot statistics about image sizes and channels."""
    # Convert sizes to numpy array for easier manipulation
    sizes = np.array(image_sizes)
    
    # Plot image sizes
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(sizes[:, 0], sizes[:, 1], alpha=0.5)
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('Image Dimensions')
    
    # Plot channel distribution
    plt.subplot(1, 2, 2)
    channel_counts = Counter(image_channels)
    plt.bar(channel_counts.keys(), channel_counts.values())
    plt.xlabel('Number of Channels')
    plt.ylabel('Count')
    plt.title('Channel Distribution')
    
    plt.tight_layout()
    plt.savefig('image_statistics.png')
    plt.close()

def get_args():
    args = ArgumentParser()
    args.add_argument('--dataset_root', type=str)
    args.add_argument('--output_name', type=str)
    return args.parse_args()

def main():
    # Set the root directory of your dataset

    args = get_args()
    root_dir = args.dataset_root  # Update this path to your dataset location
    output_name = args.output_name    
    # Analyze dataset structure
    train_counts, valid_counts = analyze_dataset_structure(root_dir)
    print("\nDataset Structure:")
    print("Training set class distribution:", train_counts)
    print("Validation set class distribution:", valid_counts)
    
    # Analyze image statistics
    image_sizes, image_channels = analyze_image_statistics(root_dir)
    print("\nImage Statistics:")
    print(f"Total number of images analyzed: {len(image_sizes)}")
    print(f"Unique image sizes: {set(image_sizes)}")
    print(f"Channel distribution: {Counter(image_channels)}")
    
    # Analyze class distribution
    analyze_class_distribution(root_dir, args.output_name)

if __name__ == "__main__":
    main() 