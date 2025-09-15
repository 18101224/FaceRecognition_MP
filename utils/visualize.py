import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import math


def crop_to_square_grid(x):
    batch = x.shape[0]
    grid_size = max(int(math.sqrt(batch)), 1)
    keep = grid_size * grid_size
    return x[:keep], grid_size
    
def calc_angle(centers, embeds):
    mat = embeds@centers.transpose(1,2)
    angles = torch.arccos(mat)
    return angles # bs, 7 ( angle based on centers)



def plot_tsne(embeddings, ys, y_hats, model_name):
    tsne=TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    fig, axes = plt.subplots(1,2,figsize=(12,5))

    scatter1 = axes[0].scatter(
        embeddings_2d[:,0],embeddings_2d[:,1], c = ys, cmap='jet',alpha=0.5,edgecolor='k'
    )
    axes[0].set_title('true_label')
    plt.colorbar(scatter1, ax=axes[0],label='True Label')

    scatter2 = axes[1].scatter(
        embeddings_2d[:,0],embeddings_2d[:,1], c=y_hats, cmap='jet', alpha=0.5, edgecolor='k'

    )
    axes[1].set_title('predicted_label')
    plt.colorbar(scatter2, ax=axes[1],label='Predicted Label')
    plt.tight_layout()
    plt.title('embedding scattered')
    plt.savefig(f'{model_name}.png')


def plot_circle(centers,embeddings,ys):
    tsne = TSNE(n_components=2, random_state=42)
    centers_2d = tsne.fit_transform(centers)
    centers_2d = centers_2d/centers_2d.norm(dim=1)
    thetas = np.arccos(centers_2d[:,0])
    angles = calc_angle(centers,embeddings)
    theta_plus = thetas[0] + embeddings[:,0]
    xs = np.cos(theta_plus)
    ys = np.sin(theta_plus)
    plt.scatter(xs,ys,c='bule')