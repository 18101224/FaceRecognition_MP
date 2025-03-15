import sys , torch
sys.path.append('../')
from Loss import get_label_noise
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


@torch.no_grad()
def compute_all_label_noise(dataloader, model, aligner, device):
    result = []
    for img, label, _, _ in tqdm(dataloader):
        img = img.to(device)
        label = label.to(device)
        _, ldmk, _, _, _, _ = aligner(img)
        _, pred = model(img, ldmk)
        j = get_label_noise(label=label, pred=pred)
        _, p = torch.max(pred, dim=-1)
        sign = p == label
        sign = sign.reshape(-1, 1)
        j = j.reshape(-1, 1)
        temp = torch.concat((j, sign), dim=-1)
        result.append(temp)
    result = torch.vstack(result)
    return result.detach().cpu().numpy()

def plot_j_distribution(j_info, save_path=None, bins=50):
    """
    Plot the distribution of J values with different colors for positive and negative signs.
    
    Args:
        j_info: numpy array or torch tensor of shape (num_samples, 2) where
               first column is J values and second column is signs (boolean)
        save_path: Optional path to save the plot
        bins: Number of bins for the histogram
    """
    if isinstance(j_info, torch.Tensor):
        j_info = j_info.cpu().numpy()
    
    j_values = j_info[:, 0]
    signs = j_info[:, 1].astype(bool)
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot histograms
    plt.hist(j_values[signs], bins=bins, alpha=0.6, color='blue', 
             label='Positive Sign', density=True)
    plt.hist(j_values[~signs], bins=bins, alpha=0.6, color='red',
             label='Negative Sign', density=True)
    
    # Add vertical line at x=0
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Add labels and title
    plt.xlabel('J Value')
    plt.ylabel('Density')
    plt.title('Distribution of J Values by Sign')
    plt.legend()
    
    # Add statistics as text
    stats_text = f"Statistics:\n"
    stats_text += f"Total samples: {len(j_values)}\n"
    stats_text += f"Positive signs: {np.sum(signs)} ({100*np.mean(signs):.1f}%)\n"
    stats_text += f"Mean J: {np.mean(j_values):.3f}\n"
    stats_text += f"Std J: {np.std(j_values):.3f}"
    
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def analyze_j_distribution(j_info):
    """
    Analyze the J distribution and print detailed statistics.
    
    Args:
        j_info: numpy array or torch tensor of shape (num_samples, 2) where
               first column is J values and second column is signs (boolean)
    """
    if isinstance(j_info, torch.Tensor):
        j_info = j_info.cpu().numpy()
    
    j_values = j_info[:, 0]
    signs = j_info[:, 1].astype(bool)
    
    print("\nJ Distribution Analysis:")
    print("-----------------------")
    print(f"Total samples: {len(j_values)}")
    print(f"Positive signs: {np.sum(signs)} ({100*np.mean(signs):.1f}%)")
    print(f"Negative signs: {np.sum(~signs)} ({100*np.mean(~signs):.1f}%)")
    print("\nOverall Statistics:")
    print(f"Mean J: {np.mean(j_values):.3f}")
    print(f"Median J: {np.median(j_values):.3f}")
    print(f"Std J: {np.std(j_values):.3f}")
    print(f"Min J: {np.min(j_values):.3f}")
    print(f"Max J: {np.max(j_values):.3f}")
    print("\nBy Sign:")
    print("Positive Sign:")
    print(f"  Mean J: {np.mean(j_values[signs]):.3f}")
    print(f"  Std J: {np.std(j_values[signs]):.3f}")
    print("Negative Sign:")
    print(f"  Mean J: {np.mean(j_values[~signs]):.3f}")
    print(f"  Std J: {np.std(j_values[~signs]):.3f}")
