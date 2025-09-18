import torch
from models import ImbalancedModel 
from dataset import get_fer_transforms, FER 
from torch.utils.data import DataLoader
import torch 
from argparse import Namespace
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import random
import torchvision.transforms as T
from tqdm import tqdm 
import os

torch.set_grad_enabled(False)


model = ImbalancedModel(num_classes=7, model_type='ir50',feature_branch=True, cos=True,learnable_input_dist=True)
ckpt = torch.load('checkpoint/pm6cce14bf-a2b1-4539-b848-151fd542c74a/latest.pth', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt)
model = model.backbone
model.eval()
model = model.cuda()

args = {'dataset_path':'../data/RAF-DB_balanced','dataset_name':'RAF-DB'}
args = Namespace(**args)
dataset = FER(args, transform=get_fer_transforms(train=False), train=True, idx=False)
loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=32, pin_memory=True)

checkpoint_path = 'checkpoint/fr_db_fer.pth'
temp_checkpoint_path = 'checkpoint/fr_db_temp.pth'

if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path)
    features = ckpt['features']
    labels = ckpt['labels']
    all_images = ckpt['all_images']
else:
    features = []
    labels = []
    all_images = []  # Store original images
    for img, label in tqdm(loader):
        img = img.cuda()
        labels.append(label)
        feature = model(img)[0]
        features.append(feature.detach().cpu())
        all_images.append(img.cpu())

    labels = torch.cat(labels, dim=0)
    features = torch.cat(features, dim=0)
    all_images = torch.cat(all_images, dim=0)
    features = torch.nn.functional.normalize(features, p=2, dim=1)

    pass

# Compute neighbor indices from saved or newly computed features/labels
# Calculate pairwise cosine similarities
similarities = torch.mm(features, features.t())  # [num_samples, num_samples]
similarities.fill_diagonal_(-float('inf'))

nearest_same_label = []  # (N, 3) global indices
most_far_same_label = []  # (N, 3) global indices
nearest_diff_label = []  # (N, 3) global indices
most_far_diff_label = []  # (N, 3) global indices

num_samples_total = labels.shape[0]
for i in range(num_samples_total):
    label_i = labels[i]
    same_indices = torch.nonzero(labels == label_i, as_tuple=False).squeeze(1)
    same_indices = same_indices[same_indices != i]
    diff_indices = torch.nonzero(labels != label_i, as_tuple=False).squeeze(1)

    same_sims = similarities[i, same_indices]
    diff_sims = similarities[i, diff_indices]

    k_same_close = min(3, same_sims.numel())
    k_same_far = min(3, same_sims.numel())
    k_diff_close = min(3, diff_sims.numel())
    k_diff_far = min(3, diff_sims.numel())

    same_close_global = same_indices[torch.topk(same_sims, k_same_close, largest=True).indices]
    same_far_global = same_indices[torch.topk(same_sims, k_same_far, largest=False).indices]
    diff_close_global = diff_indices[torch.topk(diff_sims, k_diff_close, largest=True).indices]
    diff_far_global = diff_indices[torch.topk(diff_sims, k_diff_far, largest=False).indices]

    # Ensure length 3 by padding with self index if needed (rare)
    def pad_to_three(t: torch.Tensor, fill: int) -> torch.Tensor:
        if t.numel() < 3:
            pad = t.new_full((3 - t.numel(),), fill)
            return torch.cat([t, pad], dim=0)
        return t

    nearest_same_label.append(pad_to_three(same_close_global, i))
    most_far_same_label.append(pad_to_three(same_far_global, i))
    nearest_diff_label.append(pad_to_three(diff_close_global, i))
    most_far_diff_label.append(pad_to_three(diff_far_global, i))

nearest_same_label = torch.stack(nearest_same_label)
most_far_same_label = torch.stack(most_far_same_label)
nearest_diff_label = torch.stack(nearest_diff_label)
most_far_diff_label = torch.stack(most_far_diff_label)

ckpt = {
    'nearest_same_label': nearest_same_label,
    'most_far_same_label': most_far_same_label,
    'nearest_diff_label': nearest_diff_label,
    'most_far_diff_label': most_far_diff_label,
    'features': features,
    'labels': labels,
    'all_images': all_images,
}
torch.save(ckpt, temp_checkpoint_path)
os.replace(temp_checkpoint_path, checkpoint_path)


def plot_and_save_grid(selected_indices, all_images, labels, nearest_same_label, most_far_same_label, nearest_diff_label, most_far_diff_label, filename='neighbors_grid.png'):
    num_samples = len(selected_indices)
    fig, axes = plt.subplots(num_samples, 13, figsize=(26, 2 * num_samples))
    
    for row, idx in enumerate(selected_indices):
        original_img = all_images[idx]
        original_label = labels[idx].item()

        # Helper to set edge color
        def set_edge(ax, color):
            for sp in ax.spines.values():
                sp.set_color(color)
                sp.set_linewidth(4)

        # Plot original image
        axes[row, 0].imshow(original_img.permute(1, 2, 0).numpy())
        axes[row, 0].set_title('Original')
        set_edge(axes[row, 0], 'green')

        # Plot nearest same label
        for col, img_idx in enumerate(nearest_same_label[idx].tolist(), start=1):
            img = all_images[img_idx]
            same = (labels[img_idx].item() == original_label)
            edge_color = 'green' if same else 'red'
            axes[row, col].imshow(img.permute(1, 2, 0).numpy())
            axes[row, col].set_title('Close S' if same else 'Close D', color=edge_color, fontsize=8)
            set_edge(axes[row, col], edge_color)

        # Plot most far same label
        for col, img_idx in enumerate(most_far_same_label[idx].tolist(), start=4):
            img = all_images[img_idx]
            same = (labels[img_idx].item() == original_label)
            edge_color = 'green' if same else 'red'
            axes[row, col].imshow(img.permute(1, 2, 0).numpy())
            axes[row, col].set_title('Far S' if same else 'Far D', color=edge_color, fontsize=8)
            set_edge(axes[row, col], edge_color)

        # Plot nearest different label (3)
        for col, img_idx in enumerate(nearest_diff_label[idx].tolist(), start=7):
            img = all_images[img_idx]
            same = (labels[img_idx].item() == original_label)
            edge_color = 'green' if same else 'red'
            axes[row, col].imshow(img.permute(1, 2, 0).numpy())
            axes[row, col].set_title('Close S' if same else 'Close D', color=edge_color, fontsize=8)
            set_edge(axes[row, col], edge_color)

        # Plot most far different label (3)
        for col, img_idx in enumerate(most_far_diff_label[idx].tolist(), start=10):
            img = all_images[img_idx]
            same = (labels[img_idx].item() == original_label)
            edge_color = 'green' if same else 'red'
            axes[row, col].imshow(img.permute(1, 2, 0).numpy())
            axes[row, col].set_title('Far S' if same else 'Far D', color=edge_color, fontsize=8)
            set_edge(axes[row, col], edge_color)

    for ax in axes.flatten():
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Select random samples
random_indices = np.random.choice(len(labels), 12, replace=False)
plot_and_save_grid(random_indices, all_images, labels, nearest_same_label, most_far_same_label, nearest_diff_label, most_far_diff_label)

