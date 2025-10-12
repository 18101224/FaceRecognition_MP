import matplotlib.pyplot as plt
import numpy as np
import os 
import seaborn as sns
from collections import Counter 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def plot_label_distribution(train_labels, valid_labels, save_path, model_name=None):
    """
    Plot histograms of class distribution for training and validation datasets side by side.
    Args:
        train_labels (list or np.array): Labels for the training set.
        valid_labels (list or np.array): Labels for the validation set.
        save_path (str): Path to save the histogram image.
        model_name (str, optional): Name of the model to annotate the figure.
    """
    train_labels = np.array(train_labels)
    valid_labels = np.array(valid_labels)
    classes = np.arange(max(train_labels.max(), valid_labels.max()) + 1)
    train_counts = np.bincount(train_labels, minlength=len(classes))
    valid_counts = np.bincount(valid_labels, minlength=len(classes))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(classes, train_counts, tick_label=classes)
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Train Set Class Distribution')
    axes[1].bar(classes, valid_counts, tick_label=classes, color='orange')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Validation Set Class Distribution')
    if model_name is not None:
        fig.suptitle(f'Model: {model_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96] if model_name is not None else None)
    plt.savefig(os.path.join(save_path, 'label_distribution.png'))
    plt.close()

def plot_confusion_matrix(cm: np.ndarray, normed_cm: np.ndarray, save_path: str, model_name=None, save_name=None):
    """
    Plot a confusion matrix and its normalized version side by side as heatmaps.
    Args:
        cm (np.ndarray): Confusion matrix.
        normed_cm (np.ndarray): Normalized confusion matrix.
        save_path (str): Path to save the plot.
        model_name (str, optional): Name of the model to annotate the figure.
    """
    if cm.shape[0] < 20:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        # Plot raw confusion matrix with annotation
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted Label')
        axes[0].set_ylabel('True Label')
        # Plot normalized confusion matrix with annotation
        sns.heatmap(normed_cm, annot=True, fmt='.2f', cmap='Blues', ax=axes[1])
        axes[1].set_title('Normalized Confusion Matrix')
        axes[1].set_xlabel('Predicted Label')
        axes[1].set_ylabel('True Label')
        if model_name is not None:
            fig.suptitle(f'Model: {model_name}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96] if model_name is not None else None)
        if save_name is not None:
            plt.savefig(os.path.join(save_path, save_name))
        else:
            plt.savefig(os.path.join(save_path, 'confusion_matrices.png'))
        plt.close() 
    else:
        # If the number of classes is large, plot heatmaps without annotation values for clarity
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        # Plot raw confusion matrix without annotation
        sns.heatmap(cm, annot=False, cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted Label')
        axes[0].set_ylabel('True Label')
        # Plot normalized confusion matrix without annotation
        sns.heatmap(normed_cm, annot=False, cmap='Blues', ax=axes[1])
        axes[1].set_title('Normalized Confusion Matrix')
        axes[1].set_xlabel('Predicted Label')
        axes[1].set_ylabel('True Label')
        if model_name is not None:
            fig.suptitle(f'Model: {model_name}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96] if model_name is not None else None)
        if save_name is not None:
            plt.savefig(os.path.join(save_path, save_name))
        else:
            plt.savefig(os.path.join(save_path, 'confusion_matrices.png'))
        plt.close() 

def compare_confusion_matrices(normed_cms: list, model_names: list, save_path: str, save_name: str = 'compare_confusion_matrices.png'):
    """
    Plot normalized confusion matrices for multiple models side by side for comparison.
    Args:
        normed_cms (list[np.ndarray]): List of normalized confusion matrices (one per model), each shape (n_c, n_c).
        model_names (list[str]): Names for each model, used as subplot titles.
        save_path (str): Directory to save the figure.
        save_name (str): Filename to save under.
    """
    if not isinstance(normed_cms, list) or len(normed_cms) == 0:
        return
    n_models = len(normed_cms)
    n_c = normed_cms[0].shape[0]
    annot = True if n_c <= 20 else False
    fmt = '.2f'
    fig_width = max(6, 6 * n_models)
    fig, axes = plt.subplots(1, n_models, figsize=(fig_width, 6))
    axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    for idx, (cm, ax) in enumerate(zip(normed_cms, axes)):
        sns.heatmap(cm, annot=annot, fmt=fmt, cmap='Blues', vmin=0.0, vmax=1.0, ax=ax)
        title = 'Normalized Confusion Matrix'
        if model_names is not None and idx < len(model_names):
            title += f' ({model_names[idx]})'
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, save_name))
    plt.close()

def plot_angle_with_confusion_matrix(angle: np.ndarray, conf: np.ndarray, save_path: str, save_name=None,
                                     top_reserved: float = 0.14):
    """
    Plot heatmaps for angle matrix and confusion matrix side by side.
    - Angle heatmap: base value is mean of upper triangle (excluding diagonal), use diverging colormap, mask diagonal.
    - Confusion matrix: mask diagonal, plot only off-diagonal elements.
    - No annotation if n_c > 20.
    - Also, display mean and std of upper triangle (excluding diagonal) of angle matrix as text in the figure.
    Args:
        angle (np.ndarray): Angle matrix of shape (n_c, n_c)
        conf (np.ndarray): Confusion matrix of shape (n_c, n_c)
        save_path (str): Path to save the plot.
    """
    # Defaults for angle heatmap colorbar center/range if not provided
    # Theoretical equi-angular center with margin

    bar_center = np.arccos(-1/(angle.shape[0]-1))*180.0/np.pi

    bar_range = 10 
    n_c = angle.shape[0]
    # Mask diagonal for both matrices
    mask = np.eye(n_c, dtype=bool)
    # For angle: get mean and std of upper triangle (excluding diagonal)
    triu_indices = np.triu_indices(n_c, k=1)
    upper_vals = angle[triu_indices]
    base = upper_vals.mean()
    std = upper_vals.std()
    # Prepare angle matrix for plotting (mask diagonal); we keep absolute values
    # and control the color scale using bar_center/bar_range
    angle_plot = angle.copy()
    # For confusion: mask diagonal (set to np.nan)
    conf_masked = conf.copy().astype(float)
    np.fill_diagonal(conf_masked, np.nan)
    # Plot
    if n_c <= 20:
        annot_angle = True
        annot_conf = True
        fmt_angle = '.2f'
        fmt_conf = '.2f'  # Use float format for confusion matrix since it contains np.nan
    else:
        annot_angle = False
        annot_conf = False
        fmt_angle = '.2f'
        fmt_conf = '.2f'
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    # Angle heatmap (absolute values with controlled color scale)
    sns.heatmap(
        angle_plot,
        mask=mask,
        annot=annot_angle,
        fmt=fmt_angle,
        cmap='coolwarm',
        center=bar_center,
        vmin=bar_center - bar_range,
        vmax=bar_center + bar_range,
        ax=axes[0]
    )
    axes[0].set_title('Angle Matrix (off-diagonal)')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Class')
    # Confusion matrix heatmap
    sns.heatmap(conf_masked, mask=mask, annot=annot_conf, fmt=fmt_conf, cmap='Blues', ax=axes[1])
    axes[1].set_title('Confusion Matrix (off-diagonal)')
    axes[1].set_xlabel('Predicted Class')
    axes[1].set_ylabel('True Class')
    # Build stats textbox and place it in reserved top area to avoid covering heatmap
    textstr = f"Angle mean: {base:.2f}\nAngle std: {std:.2f}"
    # Reserve top space for the textbox
    plt.tight_layout(rect=[0, 0.0, 1, 1 - top_reserved])
    fig.text(0.015, 1 - (top_reserved / 2), textstr, fontsize=13,
             va='center', ha='left',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.9))
    if save_name is not None:
        plt.savefig(os.path.join(save_path, save_name))
    else:
        plt.savefig(os.path.join(save_path, 'angle_confusion_heatmaps.png'))
    plt.close()


def plot_class_num_and_error(training_labels: list, error_rate: list, save_path: str, model_name=None, save_name=None):
    """
    Plot a curve where x-axis is the number of samples per class and y-axis is the error rate per class.
    Args:
        training_labels (list): List of class labels for the training set.
        error_rate (list): List of error rates per class (should match number of classes).
        save_path (str): Path to save the plot image.
        model_name (str, optional): Name of the model to annotate the figure.
    """
    counter = Counter(training_labels)
    num_samples = [0] * len(counter)
    for key, value in counter.items():
        num_samples[int(key)] = value
    vectors = np.hstack((np.array(num_samples).reshape(-1, 1), np.array(error_rate).reshape(-1, 1)))
    # Sort by number of samples (feature 0)
    vectors = vectors[vectors[:, 0].argsort()]
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(vectors[:, 0], vectors[:, 1], marker='o')
    plt.xlabel('Number of Samples per Class')
    plt.ylabel('Error Rate per Class')
    plt.title('Class Sample Count vs. Error Rate')
    plt.ylim(0.0, 0.6)  # Set y-axis range
    plt.grid(True)
    if model_name is not None:
        plt.suptitle(f'Model: {model_name}', fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96] if model_name is not None else None)
    if save_name is not None:
        plt.savefig(os.path.join(save_path, save_name))
    else:
        plt.savefig(os.path.join(save_path, 'class_num_vs_error.png'))
    plt.close()


def plot_class_num_and_accuracy(training_labels: list, error_rate: list, save_path: str, model_name=None, save_name=None):
    """
    Plot a bar chart where x-axis is the class index and y-axis is the accuracy per class.
    Converts error rates to accuracy by subtracting from 1.
    Args:
        training_labels (list): List of class labels for the training set.
        error_rate (list): List of error rates per class (should match number of classes).
        save_path (str): Path to save the plot image.
        model_name (str, optional): Name of the model to annotate the figure.
    """
    counter = Counter(training_labels)
    num_samples = [0] * len(counter)
    for key, value in counter.items():
        num_samples[int(key)] = value
    # Convert error rates to accuracy
    accuracy = 1.0 - np.array(error_rate)
    
    # Create class indices for x-axis
    class_indices = np.arange(len(accuracy))
    
    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_indices, accuracy, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Color bars based on sample count (optional enhancement)
    sample_counts = np.array(num_samples)
    if len(sample_counts) > 0:
        # Normalize sample counts for coloring
        norm_counts = (sample_counts - sample_counts.min()) / (sample_counts.max() - sample_counts.min() + 1e-8)
        colors = plt.cm.viridis(norm_counts)
        for bar, color in zip(bars, colors):
            bar.set_color(color)
    
    plt.xlabel('Class Index')
    plt.ylabel('Accuracy per Class')
    plt.title('Class-wise Accuracy')
    plt.ylim(0.0, 1.0)  # Set y-axis range
    plt.grid(True, alpha=0.3)
    
    # Add sample count as secondary information
    ax2 = plt.gca().twinx()
    ax2.plot(class_indices, sample_counts, 'r--', alpha=0.6, marker='o', markersize=3, linewidth=1, label='Sample Count')
    ax2.set_ylabel('Number of Samples per Class', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right')
    
    if model_name is not None:
        plt.suptitle(f'Model: {model_name}', fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96] if model_name is not None else None)
    if save_name is not None:
        plt.savefig(os.path.join(save_path, save_name))
    else:
        plt.savefig(os.path.join(save_path, 'class_num_vs_accuracy.png'))
    plt.close()


def plot_error_rate_comparison(error_rates_list, model_names, train_labels, save_path: str):
    """
    Plot class-wise error rates for multiple models as grouped bars, and normalized sample counts as a line.
    Args:
        error_rates_list (list of array-like): List of error rate lists, one per model (each of length = num_classes)
        model_names (list of str): List of model names (length = number of models)
        train_labels (array-like): List of training labels to compute sample counts per class
        save_path (str): Path to save the plot image
    """
    error_rates_list = [np.array(er) for er in error_rates_list]
    train_labels = np.array(train_labels)
    num_models = len(error_rates_list)
    num_classes = len(error_rates_list[0])
    class_labels = [f'class {i}' for i in range(num_classes)]
    # Compute sample counts per class
    counter = Counter(train_labels)
    sample_counts = np.array([counter.get(i, 0) for i in range(num_classes)])
    # Normalize sample counts (0~1)
    sample_counts_ratio = sample_counts / sample_counts.max() if sample_counts.max() > 0 else sample_counts

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(num_classes)
    total_bar_width = 0.7
    bar_width = total_bar_width / num_models
    # Plot error rate bars for each model
    for idx, (error_rate, model_name) in enumerate(zip(error_rates_list, model_names)):
        offset = -total_bar_width/2 + idx * bar_width + bar_width/2
        ax1.bar(x + offset, error_rate, width=bar_width, label=model_name, alpha=0.7)
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Error Rate')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_labels, rotation=45, ha='right')
    ax1.set_ylim(0, 1)
    ax1.legend(loc='upper left')
    # Normalized sample count (right y-axis)
    ax2 = ax1.twinx()
    ax2.plot(x, sample_counts_ratio, color='black', marker='o', linestyle='--', linewidth=2, label='Sample Counts Ratio')
    ax2.set_ylabel('Normalized Sample Count (Ratio)')
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc='upper right')
    plt.title('Class-wise Error Rate Comparison with Normalized Training Sample Counts')
    fig.tight_layout()
    # Add extra space for x labels and title
    plt.subplots_adjust(bottom=0.18, top=0.88)
    plt.savefig(os.path.join(save_path, 'error_rate_comparison.png'))
    plt.close()

def plot_accuracy_comparison(accuracy_list, model_names, train_labels, save_path: str, save_name=None):
    """
    Plot class-wise accuracies for multiple models as grouped bars, and normalized sample counts as a line.
    Args:
        accuracy_list (list of array-like): List of accuracy lists, one per model (each of length = num_classes)
        model_names (list of str): List of model names (length = number of models)
        train_labels (array-like): List of training labels to compute sample counts per class
        save_path (str): Path to save the plot image
    """
    accuracy_list = [np.array(acc) for acc in accuracy_list]
    train_labels = np.array(train_labels)
    num_models = len(accuracy_list)
    num_classes = len(accuracy_list[0])

    # Compute sample counts per class
    counter = Counter(train_labels)
    sample_counts = np.array([counter.get(i, 0) for i in range(num_classes)])

    # Sort classes by sample count in descending order
    sorted_indices = np.argsort(sample_counts)[::-1]  # Sort in descending order
    sorted_sample_counts = sample_counts[sorted_indices]
    sorted_accuracies = [acc[sorted_indices] for acc in accuracy_list]
    sorted_class_labels = [f'class {i} ({sample_counts[i]})' for i in sorted_indices]

    # Normalize sample counts (0~1)
    sample_counts_ratio = sorted_sample_counts / sorted_sample_counts.max() if sorted_sample_counts.max() > 0 else sorted_sample_counts

    fig, ax1 = plt.subplots(figsize=(12, 6))
    x = np.arange(num_classes)
    total_bar_width = 0.7
    bar_width = total_bar_width / num_models

    # Plot accuracy bars for each model
    for idx, (acc, model_name) in enumerate(zip(sorted_accuracies, model_names)):
        offset = -total_bar_width/2 + idx * bar_width + bar_width/2
        ax1.bar(x + offset, acc, width=bar_width, label=model_name, alpha=0.7)
    ax1.set_xlabel('Class (sorted by sample count descending)')
    ax1.set_ylabel('Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sorted_class_labels, rotation=45, ha='right')
    ax1.set_ylim(0, 1)
    ax1.legend(loc='upper left')

    # Normalized sample count (right y-axis)
    ax2 = ax1.twinx()
    ax2.plot(x, sample_counts_ratio, color='black', marker='o', linestyle='--', linewidth=2, label='Sample Counts Ratio')
    ax2.set_ylabel('Normalized Sample Count (Ratio)')
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc='upper right')

    plt.title('Class-wise Accuracy Comparison with Normalized Training Sample Counts\n(Sorted by Sample Count Descending)')
    fig.tight_layout()
    # Add extra space for x labels and title
    plt.subplots_adjust(bottom=0.25, top=0.85)
    plt.savefig(os.path.join(save_path, 'accuracy_comparison.png' if save_name is None else save_name))
    plt.close()

def compare_angle_rates(angle_matrices, model_names, save_path: str):
    """
    Compare multiple angle matrices by plotting them side by side, centered by the mean of the matrix with the lowest std.
    Display each matrix's mean and std in a text box on its subplot, and annotate with model name.
    Args:
        angle_matrices (list of np.ndarray): List of angle matrices (n_c, n_c)
        model_names (list of str): List of model names
        save_path (str): Path to save the plot
    """
    n_models = len(angle_matrices)
    n_c = angle_matrices[0].shape[0]
    mask = np.eye(n_c, dtype=bool)
    # Compute upper triangle (excluding diagonal) mean and std for all
    triu_indices = np.triu_indices(n_c, k=1)
    uppers = [mat[triu_indices] for mat in angle_matrices]
    means = [vals.mean() for vals in uppers]
    stds = [vals.std() for vals in uppers]
    # Choose base mean from matrix with lowest std
    base = np.arccos(-1/(n_c-1))*180.0/np.pi
    # Center all matrices by the chosen base
    centered_matrices = [mat - base for mat in angle_matrices]
    # Plot
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 7))
    if n_models == 1:
        axes = [axes]
    for idx, (mat, mean, std, ax) in enumerate(zip(centered_matrices, means, stds, axes)):
        sns.heatmap(mat, mask=mask, annot=(n_c <= 20), fmt='.2f', cmap='coolwarm', center=0, ax=ax)
        title = f'Angle Matrix (centered)'
        if model_names is not None and idx < len(model_names):
            title += f'\n{model_names[idx]}'
        ax.set_title(title)
        ax.set_xlabel('Class')
        ax.set_ylabel('Class')
        # Add mean and std text box in upper left
        textstr = f"mean: {mean:.2f}\nstd: {std:.2f}"
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=13,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.9))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(save_path, 'compare_angle_rates.png'))
    plt.close()

def plot_tsne_features(features, labels, train_labels, save_path: str, original_features=None, model_name=None):
    """
    Plot t-SNE features with color mapped to class sample count.
    Also plot the center of mass for each class.
    If original_features is provided, return the centers in the original feature space.
    Least frequent class is yellow, most frequent is purple, others are linearly interpolated.
    Args:
        features (np.ndarray): t-SNE features, shape (n_samples, 2) or (n_samples, output_dim)
        labels (np.ndarray): integer class labels, shape (n_samples,)
        save_path (str): path to save the figure
        original_features (np.ndarray or None): original feature vectors, shape (n_samples, original_dim)
        model_name (str, optional): Name of the model to annotate the figure.
    Returns:
        np.ndarray or None: class centers in original feature space, shape (num_classes, original_dim), or None if not provided
    """
    features = np.asarray(features)
    labels = np.asarray(labels)
    assert features.shape[0] == labels.shape[0], "Features and labels must have the same number of samples."
    # Count label distribution
    label_counts = Counter(train_labels)
    classes = sorted(list(label_counts.keys()))
    counts = np.array([label_counts[c] for c in classes])
    # Map each class to a color based on its count
    min_count, max_count = counts.min(), counts.max()
    print(min_count, max_count)
    norm = plt.Normalize(vmin=min_count, vmax=max_count)
    cmap = plt.get_cmap('plasma')  # yellow (low) to purple (high)
    class_to_color = {c: cmap(norm(label_counts[c])) for c in classes}
    # Assign color to each sample
    sample_colors = np.array([class_to_color[l] for l in labels])
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(features[:, 0], features[:, 1], c=sample_colors, s=10, edgecolor='none')
    # Compute and plot class centers
    original_centers = [] if original_features is not None else None
    for c in classes:
        class_mask = (labels == c)
        class_features = features[class_mask]
        center = class_features.mean(axis=0)
        ax.scatter(center[0], center[1], c=[class_to_color[c]], s=120, marker='o', edgecolor='black', linewidths=1.5, label=f'Class {c} center')
        if original_features is not None:
            orig_center = np.asarray(original_features)[class_mask].mean(axis=0)
            orig_center_norm = np.linalg.norm(orig_center)
            if orig_center_norm > 0:
                orig_center = orig_center / orig_center_norm
            original_centers.append(orig_center)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('t-SNE Feature Distribution by Class Sample Count')
    # Create a colorbar for class sample count
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Class Sample Count')
    # Annotate colorbar with min/max class
    min_class = classes[np.argmin(counts)]
    max_class = classes[np.argmax(counts)]
    cbar.ax.text(1.1, 0, f'Least: {min_class} ({min_count})', va='bottom', ha='left', fontsize=10, color='black', transform=cbar.ax.transAxes)
    cbar.ax.text(1.1, 1, f'Most: {max_class} ({max_count})', va='top', ha='left', fontsize=10, color='black', transform=cbar.ax.transAxes)
    if model_name is not None:
        fig.suptitle(f'Model: {model_name}', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96] if model_name is not None else None)
    fig.savefig(os.path.join(save_path, 'tsne_distribution.png'))
    plt.close(fig)
    if original_features is not None:
        return np.stack(original_centers, axis=0)
    else:
        return None
    

def plot_angle_matrix(angle_matrices, save_path: str, model_names=None):
    """
    Plot one or more angle matrices as heatmaps side by side.
    If n_c <= 20, show the angle values in the heatmap and display mean/std of upper triangle (excluding diagonal) in a box at the upper right.
    If multiple matrices, center all heatmaps at the mean (m) of the matrix with the smallest std (s), and use color range (m-4*s, m+4*s).
    If abs_statistics is provided as (mean, std), use mean as base and range [mean-std, mean+std].
    Args:
        angle_matrices (np.ndarray or list of np.ndarray): Angle matrix/matrices of shape (n_c, n_c)
        save_path (str): Path to save the plot
        model_names (list of str or None): Optional list of model names for annotation
        abs_statistics (tuple or None): (mean, std) to use as base and range
    """
    
    abs_statistics = (np.arccos(-1/(angle_matrices[0].shape[0]-1))*180.0/np.pi, 10)
    # Handle single matrix input
    if isinstance(angle_matrices, np.ndarray):
        angle_matrices = [angle_matrices]
    n_models = len(angle_matrices)
    n_c = angle_matrices[0].shape[0]
    # Compute stats for all matrices
    triu_indices = np.triu_indices(n_c, k=1)
    uppers = [mat[triu_indices] for mat in angle_matrices]
    means = [vals.mean() for vals in uppers]
    stds = [vals.std() for vals in uppers]
    # Determine base and range

    m, s = abs_statistics
    vmin = m - s
    vmax = m + s

    # Plot
    if n_models == 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 6))
    if n_models == 1:
        angle_matrices = [angle_matrices[0]]
        model_names = [model_names[0]] if model_names is not None else None
    for idx, (mat, mean, std, ax) in enumerate(zip(angle_matrices, means, stds, axes)):
        m_centered = mat - m
        np.fill_diagonal(m_centered, np.nan)
        annot = True if n_c <= 20 else False
        fmt = '.2f' if n_c <= 20 else None
        sns.heatmap(m_centered + m, annot=annot, fmt=fmt, cmap='coolwarm', center=m, vmin=vmin, vmax=vmax, ax=ax)
        title = 'Angle Matrix'
        if model_names is not None and idx < len(model_names):
            title += f' ({model_names[idx]})'
        ax.set_title(title)
        ax.set_xlabel('Class')
        ax.set_ylabel('Class')
        if n_c <= 20:
            textstr = f"Angle mean: {mean:.2f}\nAngle std: {std:.2f}"
            # Place annotation box in upper right
            ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=13,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.9))
    # Build a figure-level summary textbox with means/stds and place it in spare space on the right
    stats_lines = []
    for idx, (mean, std) in enumerate(zip(means, stds)):
        name = None
        if model_names is not None and idx < len(model_names):
            name = model_names[idx]
        label = f"{name}:" if name is not None else f"Matrix {idx+1}:"
        stats_lines.append(f"{label} m={mean:.2f}, s={std:.2f}")
    if len(stats_lines) > 0:
        stats_text = "\n".join(stats_lines)
        # Reserve space on the right for the figure textbox
        fig.tight_layout(rect=[0, 0, 0.82, 1])
        fig.text(0.985, 0.5, stats_text, ha='right', va='center', fontsize=12,
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.6', alpha=0.9))
    else:
        fig.tight_layout()
    if 'png' or 'jpeg' in save_path:
        fig.savefig(save_path)
    else:
        fig.savefig(os.path.join(save_path, 'angle_matrix.png'))
    plt.close(fig)
    

def plot_weight_and_centers(centers, weight, model_name, save_path):
    """
    Plot the angle matrix between class centers and classifier weights.
    Args:
        centers (np.ndarray): (num_classes, dim), normalized.
        weight (np.ndarray): (num_classes, dim), normalized.
        model_name (str): Name of the model for annotation.
        save_path (str): Directory to save the plot.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    # Compute cosine similarity matrix
    sim_matrix = centers @ weight.T  # (num_classes, num_classes)
    # Clamp for numerical stability
    sim_matrix = np.clip(sim_matrix, -1.0, 1.0)
    # Convert to angle in degrees
    angle_matrix = np.arccos(sim_matrix) * 180.0 / np.pi
    n_c = angle_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(8, 6))
    annot = True if n_c <= 20 else False
    fmt = '.2f' if n_c <= 20 else None
    sns.heatmap(angle_matrix, annot=annot, fmt=fmt, cmap='coolwarm', ax=ax, vmin=0, vmax=180)
    ax.set_title('Angle Matrix: Centers vs. Weights')
    ax.set_xlabel('Weight Class')
    ax.set_ylabel('Center Class')
    if model_name is not None:
        fig.suptitle(f'Model: {model_name}', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96] if model_name is not None else None)
    if 'png' in save_path:
        fig.savefig(save_path)
    else:   
        fig.savefig(os.path.join(save_path, 'center_weight_angle_matrix.png'))
    plt.close(fig) 

def plot_feature_distribution(features:np.ndarray, labels:np.ndarray, W:np.ndarray, train:bool, save_path:str):
    '''
    Plot cosine similarity distributions between features and classifier weights per class.
    - features : (num_samples, feature_dim)
    - labels   : (num_samples,)
    - W        : (num_classes, feature_dim)
    Saves one figure per class with x-range (-1, 1), vertical helper lines at cos(90°), cos(100°), cos(110°),
    and stacked bars colored by correctness of prediction.
    '''
    n_c = len(W)
    for i in range(n_c):
        class_mask = (labels == i)
        class_features = features[class_mask]

        # Compute logits/similarities to all classes and similarity to true class i
        preds_matrix = class_features @ W.T  # shape: (num_class_samples, n_c)
        sims = preds_matrix[:, i].astype(np.float64).flatten()
        # Predicted class per sample
        pred_labels = preds_matrix.argmax(axis=1)
        correct_indices = (pred_labels == i)

        # Numerical stability and range control for cosine similarities
        sims = np.clip(sims, -1.0, 1.0)

        # Prepare histogram bins across [-1, 1]
        bins = np.linspace(-1.0, 1.0, 41)  # 40 bins of width 0.05
        sims_correct = sims[correct_indices]
        sims_incorrect = sims[~correct_indices]

        plt.figure(figsize=(10, 6))
        # Stacked histogram: correct (green) on bottom, incorrect (red) on top
        plt.hist([sims_correct, sims_incorrect], bins=bins, stacked=True,
                 color=['tab:green', 'tab:red'], label=['Correct', 'Incorrect'], edgecolor='black', alpha=0.8)

        # Helper vertical lines at cosine values corresponding to 90°, 100°, 110°
        degs = [10, 20, 30, 40 , 50, 60, 70, 80, 90, 100, 110]
        cos_vals = [np.cos(np.deg2rad(d)) for d in degs]
        for d, c in zip(degs, cos_vals):
            plt.axvline(x=c, color='tab:blue', linestyle='--', linewidth=1.5)
            plt.text(c, plt.gca().get_ylim()[1]*0.95, f'{d}°', color='tab:blue', rotation=90,
                     ha='right', va='top', fontsize=9)

        plt.xlim(-1.0, 1.0)
        plt.xlabel('Cosine similarity (feature · weight)')
        plt.ylabel('Count')
        title_split = 'Train' if train else 'Valid'
        plt.title(f'{title_split}: Cosine Similarity Distribution for Class {i}')
        plt.legend(loc='upper left')

        # Stats textbox (on cosine similarities)
        if sims.size > 0:
            mean_sim = float(np.mean(sims))
            std_sim = float(np.std(sims))
            textstr = f'Mean: {mean_sim:.3f}\nStd: {std_sim:.3f}\nN: {sims.size}\nCorrect: {int(correct_indices.sum())}\nIncorrect: {int((~correct_indices).sum())}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.text(0.98, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                     verticalalignment='top', horizontalalignment='right', bbox=props)

        plt.tight_layout()
        temp_save_path = save_path + f'_class_{i}.png'
        plt.savefig(temp_save_path)
        plt.close()
        


def visualize_neural_collapse(
    # Required inputs (order matters):
    X_tr,          # np.ndarray or torch.Tensor, shape (N_tr, D): TRAIN feature vectors (penultimate/features)
    Z_tr,          # np.ndarray or torch.Tensor, shape (N_tr, C): TRAIN logits (pre-softmax), kept for completeness
    y_tr,          # np.ndarray or torch.Tensor, shape (N_tr,): TRAIN integer labels in [0, C-1] (argmax(Z) -> label assumed)
    X_va,          # np.ndarray or torch.Tensor, shape (N_va, D): VALID feature vectors
    Z_va,          # np.ndarray or torch.Tensor, shape (N_va, C): VALID logits (pre-softmax), used to decide misclassification
    y_va,          # np.ndarray or torch.Tensor, shape (N_va,): VALID integer labels in [0, C-1]

    # Optional classifier params (for extended NC checks; not required for plotting):
    W=None,        # np.ndarray or torch.Tensor, shape (C, D): last-layer weights (rows per class); plotted if provided
    b=None,        # np.ndarray or torch.Tensor, shape (C,): last-layer bias (unused for plotting)

    *,  # plotting/analysis options
    metric='cosine',              # fallback nearest-class-mean metric if logits not provided/usable: 'cosine' or 'euclidean'
    svd_center=True,              # center features before SVD
    tsne_perplexity=30,
    tsne_random_state=42,
    max_pca_dim=50,
    figsize=(14, 12),
    dpi=300,                      # high resolution
    savepath=None                 # dir or prefix; saves as dir/{tsne,svd,angles}.png if dir else {prefix}_{tsne,svd,angles}.png
):
    """
    Inputs (required):
      - X_tr: (N_tr, D) train features.
      - Z_tr: (N_tr, C) train logits (pre-softmax), kept for completeness.
      - y_tr: (N_tr,) train labels in [0, C-1].
      - X_va: (N_va, D) valid features.
      - Z_va: (N_va, C) valid logits (pre-softmax), used for argmax prediction.
      - y_va: (N_va,) valid labels in [0, C-1].

    Inputs (optional):
      - W: (C, D) classifier weights; if given, included in t-SNE as class-colored triangles for NC3-style alignment visualization.
      - b: (C,) classifier bias; not used in plots.

    What it does:
      1) Dimensional collapse: SVD on centered train features to compute explained-variance ratio and energy of top (C-1) components.
      2) t-SNE scatter:
         - Train points: colored by class, semi-transparent.
         - Valid points: colored by true class, opaque; misclassified (by argmax of Z_va) marked with 'x'.
         - Class means: per-class means for train and valid, emphasized as large ring markers.
         - Optional W rows: per-class triangles in the same color.
      3) Class clustering table (angles in degrees):
         - For each class, compute angles between points and their set’s mean (train→train mean, valid→valid mean), then report mean/std and counts.

    Notes:
      - Angle θ between vectors a, b follows cos θ = (a·b)/(||a||||b||); for unit vectors, θ = arccos(clip(a·b, -1, 1)) [radians] → degrees [web:111][web:91][web:100].
      - Matplotlib Axes.table is used to render the per-class table within a figure [web:80][web:81].
      - t-SNE perplexity must be < n_samples and typically in [5, 50] [web:13][web:7].
    """

    # -------------------- utils --------------------
    def to_numpy(x):
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        return None if x is None else np.asarray(x)

    def l2_normalize(Z, axis=1, eps=1e-12):
        nrm = np.linalg.norm(Z, axis=axis, keepdims=True)
        return Z / np.clip(nrm, eps, None)

    def class_means(X, y, classes_):
        means = []
        for c in classes_:
            idx = (y == c)
            means.append(X[idx].mean(axis=0) if idx.any() else np.full(X.shape[1], np.nan))
        return np.vstack(means)

    def angles_deg_to_mean(X, mean_vec):
        # normalize to unit vectors
        Xn = l2_normalize(X)
        mn = mean_vec / max(np.linalg.norm(mean_vec), 1e-12)
        # cosines and safe arccos in radians, then degrees
        cosv = np.clip(Xn @ mn, -1.0, 1.0)
        ang_rad = np.arccos(cosv)
        ang_deg = np.degrees(ang_rad)
        return ang_deg
    # ------------------------------------------------

    # Convert to numpy
    X_tr = to_numpy(X_tr).astype(np.float64)
    Z_tr = None if Z_tr is None else to_numpy(Z_tr).astype(np.float64)
    y_tr = to_numpy(y_tr).astype(int)
    X_va = to_numpy(X_va).astype(np.float64)
    Z_va = None if Z_va is None else to_numpy(Z_va).astype(np.float64)
    y_va = to_numpy(y_va).astype(int)
    if W is not None:
        W = to_numpy(W).astype(np.float64)

    # Basic checks
    assert X_tr.ndim == 2 and X_va.ndim == 2, "Features must be (N, D)"
    assert y_tr.ndim == 1 and y_va.ndim == 1, "Labels must be (N,)"
    if Z_tr is not None:
        assert Z_tr.ndim == 2 and Z_tr.shape[0] == X_tr.shape[0], "Z_tr must be (N_tr, C)"
    if Z_va is not None:
        assert Z_va.ndim == 2 and Z_va.shape[0] == X_va.shape[0], "Z_va must be (N_va, C)"

    classes = np.unique(y_tr)
    C = len(classes)
    D = X_tr.shape[1]

    # 1) SVD on train features (global-centered)
    Xs = X_tr - X_tr.mean(axis=0, keepdims=True) if svd_center else X_tr
    U, S, Vh = np.linalg.svd(Xs, full_matrices=False)
    n = Xs.shape[0]
    expl_var = (S ** 2) / max(n - 1, 1)
    expl_var_ratio = expl_var / expl_var.sum() if expl_var.sum() > 0 else expl_var
    k = min(C - 1, expl_var_ratio.size) if C >= 2 else 1
    energy_top_cminus1 = float(expl_var_ratio[:k].sum())

    # 2) Class means (train, valid)
    train_means = class_means(X_tr, y_tr, classes)
    valid_means = class_means(X_va, y_va, classes)

    # 3) Validation predictions: prefer logits argmax; fallback to NCM
    if (Z_va is not None) and (Z_va.shape[1] >= C):
        y_va_pred = np.argmax(Z_va, axis=1)
    else:
        means_for_pred = train_means.copy()
        if metric == 'cosine':
            X_pred = l2_normalize(X_va)
            M_pred = l2_normalize(means_for_pred)
            sims = X_pred @ M_pred.T
            y_va_pred = classes[np.argmax(sims, axis=1)]
        else:
            x2 = (X_va ** 2).sum(axis=1, keepdims=True)
            m2 = (means_for_pred ** 2).sum(axis=1)[None, :]
            dots = X_va @ means_for_pred.T
            d2 = x2 + m2 - 2 * dots
            y_va_pred = classes[np.argmin(d2, axis=1)]
    mis_va = (y_va_pred != y_va)

    # 4) t-SNE embedding set: train, valid, means, optional W
    blocks = [X_tr, X_va, train_means, valid_means]
    has_W = (W is not None)
    if has_W:
        assert W.ndim == 2 and W.shape[1] == D, "W must be (C, D)"
        W_norm = np.linalg.norm(W, axis=1, keepdims=True)
        W_safe = W / np.clip(W_norm, 1e-12, None)  # direction-only
        blocks.append(W_safe)

    X_concat = np.vstack(blocks)
    n_tr, n_va = len(X_tr), len(X_va)
    idx_means_tr = np.arange(n_tr + n_va, n_tr + n_va + C)
    idx_means_va = np.arange(n_tr + n_va + C, n_tr + n_va + 2 * C)
    if has_W:
        idx_W_start = n_tr + n_va + 2 * C
        idx_W = np.arange(idx_W_start, idx_W_start + W.shape[0])

    # Standardize -> optional PCA -> t-SNE
    Xz = StandardScaler(with_mean=True, with_std=True).fit_transform(X_concat)
    if Xz.shape[1] > max_pca_dim:
        pca = PCA(n_components=max_pca_dim, random_state=tsne_random_state)
        Xz = pca.fit_transform(Xz)

    N_all = Xz.shape[0]
    effective_perp = min(tsne_perplexity, max(5, min(50, N_all - 1)))
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perp,
        init='pca',
        learning_rate='auto',
        random_state=tsne_random_state,
        method='barnes_hut'
    )
    Y = tsne.fit_transform(Xz)

    # Split back
    Y_tr = Y[:n_tr]
    Y_va = Y[n_tr:n_tr + n_va]
    Y_means_tr = Y[idx_means_tr]
    Y_means_va = Y[idx_means_va]
    if has_W:
        Y_W = Y[idx_W]

    # 5) Plotting t-SNE
    from matplotlib.lines import Line2D
    cmap = plt.cm.get_cmap('tab10', min(10, C)) if C <= 10 else plt.cm.get_cmap('tab20', min(20, C))
    color_map = {c: cmap(i % cmap.N) for i, c in enumerate(classes)}

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Train points (semi-transparent)
    for c in classes:
        idx = (y_tr == c)
        if idx.any():
            ax.scatter(Y_tr[idx, 0], Y_tr[idx, 1],
                       s=12, c=[color_map[c]], alpha=0.25, marker='o', linewidths=0)

    # Valid points: correct circles
    for c in classes:
        idx = (y_va == c) & (~mis_va)
        if idx.any():
            ax.scatter(Y_va[idx, 0], Y_va[idx, 1],
                       s=22, c=[color_map[c]], alpha=1.0, marker='o',
                       edgecolors='white', linewidths=0.5)

    # Misclassified: 'x' colored by true label
    if mis_va.any():
        ax.scatter(Y_va[mis_va, 0], Y_va[mis_va, 1],
                   s=36, c=[color_map[cl] for cl in y_va[mis_va]],
                   alpha=1.0, marker='x', linewidths=1.2)

    # Class means: train circle (ring), valid square (ring)
    for i, c in enumerate(classes):
        ax.scatter(Y_means_tr[i, 0], Y_means_tr[i, 1],
                   s=260, facecolors='none', edgecolors='k', linewidths=1.6, marker='o')
        ax.scatter(Y_means_va[i, 0], Y_means_va[i, 1],
                   s=220, facecolors='none', edgecolors='k', linewidths=1.6, marker='s')

    # Optional W rows as triangles
    if has_W:
        for i, c in enumerate(classes):
            if i < W.shape[0]:
                ax.scatter(Y_W[i, 0], Y_W[i, 1],
                           s=280, c=[color_map[c]], marker='^',
                           edgecolors='k', linewidths=1.3, alpha=1.0)

    legend_elems = [
        Line2D([0], [0], marker='o', color='none', label='Train (semi-transparent)',
               markerfacecolor='gray', alpha=0.25, markersize=6, markeredgewidth=0),
        Line2D([0], [0], marker='o', color='none', label='Valid (correct)',
               markerfacecolor='gray', alpha=1.0, markeredgecolor='white', markeredgewidth=0.5, markersize=7),
        Line2D([0], [0], marker='x', color='gray', label='Valid (misclassified)',
               markersize=7, linewidth=1.2),
        Line2D([0], [0], marker='o', color='k', label='Train class mean',
               markerfacecolor='none', markersize=10, markeredgewidth=1.6),
        Line2D([0], [0], marker='s', color='k', label='Valid class mean',
               markerfacecolor='none', markersize=9, markeredgewidth=1.6),
    ]
    if has_W:
        legend_elems.append(Line2D([0], [0], marker='^', color='k', label='Classifier W (rows)',
                                   markerfacecolor='none', markersize=10, markeredgewidth=1.3))
    ax.legend(handles=legend_elems, loc='best', frameon=True)

    ax.set_title("t-SNE of Train/Valid Features with Class Means + W")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, alpha=0.15)

    # SVD scree
    fig_svd, ax_svd = plt.subplots(figsize=(8, 4), dpi=dpi)
    ax_svd.plot(np.arange(1, len(expl_var_ratio) + 1), expl_var_ratio, '-o', ms=3)
    ax_svd.axvline(x=k, color='r', linestyle='--', linewidth=1.0)
    ax_svd.set_title(f"SVD explained variance ratio (top C-1={k} sum={energy_top_cminus1:.3f})")
    ax_svd.set_xlabel("component")
    ax_svd.set_ylabel("explained variance ratio")
    ax_svd.grid(True, alpha=0.2)

    # 6) Per-class clustering angles (degrees) table
    rows = []
    cluster = {
        "per_class": [],
        "columns": ["class", "N_tr", "mean_deg_tr", "std_deg_tr", "N_va", "mean_deg_va", "std_deg_va"]
    }
    for i, c in enumerate(classes):
        # train
        idx_tr = (y_tr == c)
        ntr = int(idx_tr.sum())
        if ntr > 0 and np.all(np.isfinite(train_means[i])):
            ang_tr = angles_deg_to_mean(X_tr[idx_tr], train_means[i])
            mtr = float(np.nanmean(ang_tr))
            str_ = float(np.nanstd(ang_tr, ddof=0))
        else:
            mtr, str_, ang_tr = np.nan, np.nan, np.array([])
        # valid
        idx_va = (y_va == c)
        nva = int(idx_va.sum())
        if nva > 0 and np.all(np.isfinite(valid_means[i])):
            ang_va = angles_deg_to_mean(X_va[idx_va], valid_means[i])
            mva = float(np.nanmean(ang_va))
            sva = float(np.nanstd(ang_va, ddof=0))
        else:
            mva, sva, ang_va = np.nan, np.nan, np.array([])
        rows.append([str(c), f"{ntr}", f"{mtr:.2f}", f"{str_:.2f}", f"{nva}", f"{mva:.2f}", f"{sva:.2f}"])
        cluster["per_class"].append({
            "class": int(c),
            "N_tr": ntr, "mean_deg_tr": mtr, "std_deg_tr": str_,
            "N_va": nva, "mean_deg_va": mva, "std_deg_va": sva
        })

    # Render table figure
    fig_tbl, ax_tbl = plt.subplots(figsize=(10, 0.6 + 0.35*max(1, len(rows))), dpi=dpi)
    ax_tbl.axis('off')
    table = ax_tbl.table(
        cellText=rows,
        colLabels=cluster["columns"],
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    ax_tbl.set_title("Per-class angular clustering (degrees): mean ± std", pad=8)

    # Save
    if savepath:
        if os.path.isdir(savepath) or str(savepath).endswith(('/', '\\')):
            tsne_path = os.path.join(savepath, "tsne.png")
            svd_path = os.path.join(savepath, "svd.png")
            tbl_path = os.path.join(savepath, "angles.png")
        else:
            tsne_path = f"{savepath}_tsne.png"
            svd_path = f"{savepath}_svd.png"
            tbl_path = f"{savepath}_angles.png"
        fig.savefig(tsne_path, bbox_inches='tight')
        fig_svd.savefig(svd_path, bbox_inches='tight')
        fig_tbl.savefig(tbl_path, bbox_inches='tight')

    # Optional alignment summary if W provided
    align = None
    if has_W:
        W_dir = W_safe  # unit-normalized rows
        TM_dir = l2_normalize(train_means)
        m = min(W_dir.shape[0], TM_dir.shape[0])
        cos = (W_dir[:m] * TM_dir[:m]).sum(axis=1)
        align = {
            "cosine_W_trainMean_per_class": cos,
            "mean_cosine": float(np.nanmean(cos)),
        }

    return {
        "svd": {
            "singular_values": S,
            "explained_variance_ratio": expl_var_ratio,
            "energy_top_C_minus_1": energy_top_cminus1,
            "num_classes": int(C),
            "feature_dim": int(D),
            "centered": bool(svd_center),
            "k_C_minus_1": int(k),
        },
        "pred": {
            "y_va_pred_used": y_va_pred,
            "misclassified_mask": mis_va,
        },
        "cluster": cluster,  # per-class angular stats
        "align": align,
        "figures": {"tsne": fig, "svd": fig_svd, "angles": fig_tbl}
    }