import matplotlib.pyplot as plt
import numpy as np
import os 
import seaborn as sns
from collections import Counter 

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
    Plot a curve where x-axis is the number of samples per class and y-axis is the accuracy per class.
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
    vectors = np.hstack((np.array(num_samples).reshape(-1, 1), accuracy.reshape(-1, 1)))
    # Sort by number of samples (feature 0)
    vectors = vectors[vectors[:, 0].argsort()]
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(vectors[:, 0], vectors[:, 1], marker='o')
    plt.xlabel('Number of Samples per Class')
    plt.ylabel('Accuracy per Class')
    plt.title('Class Sample Count vs. Accuracy')
    plt.ylim(0.0, 1.0)  # Set y-axis range
    plt.grid(True)
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

def plot_accuracy_comparison(accuracy_list, model_names, train_labels, save_path: str):
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

    # Plot accuracy bars for each model
    for idx, (acc, model_name) in enumerate(zip(accuracy_list, model_names)):
        offset = -total_bar_width/2 + idx * bar_width + bar_width/2
        ax1.bar(x + offset, acc, width=bar_width, label=model_name, alpha=0.7)
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Accuracy')
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

    plt.title('Class-wise Accuracy Comparison with Normalized Training Sample Counts')
    fig.tight_layout()
    # Add extra space for x labels and title
    plt.subplots_adjust(bottom=0.18, top=0.88)
    plt.savefig(os.path.join(save_path, 'accuracy_comparison.png'))
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
    min_std_idx = np.argmin(stds)
    base = means[min_std_idx]
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
    

def plot_angle_matrix(angle_matrices, save_path: str, model_names=None, dataset_name=None):
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
    
    statistics_dict = {
        'cifar10': (np.arccos(-1/9)*180.0/np.pi,10),
        'cifar100': (np.arccos(-1/99)*180.0/np.pi,10),
        'imagenet_lt': (np.arccos(-1/999)*180.0/np.pi,10),
        'RAF-DB': (np.arccos(-1/6)*180.0/np.pi, 10),
        'AffectNet': (np.arccos(-1/6)*180.0/np.pi, 10)
    }
    abs_statistics = statistics_dict[dataset_name]
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
    if abs_statistics is not None and len(abs_statistics) == 2:
        m, s = abs_statistics
        vmin = m - s
        vmax = m + s
    else:
        # Find matrix with smallest std
        min_std_idx = int(np.argmin(stds))
        m = means[min_std_idx]
        s = stds[min_std_idx]
        vmin = m - 4 * s
        vmax = m + 4 * s
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
    if 'png' in save_path:
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
        
