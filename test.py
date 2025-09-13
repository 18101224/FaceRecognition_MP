from models import model_dict  
from dataset import get_transform 
from argparse import Namespace
from glob import glob 
import torch 
import numpy as np
from sklearn.preprocessing import normalize
import random
from PIL import Image
import matplotlib.pyplot as plt
import os
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from balanced_kmeans import kmeans_equal


device = torch.device('cuda')
model = model_dict['ir50']().to(device)
model.eval()
args = {
    'dataset_name': 'RAF-DB', 'model_type': 'ir50'
}
args = Namespace(**args)
transform = get_transform(args, train=False)


img_paths = []
img_labels = []  # New list to store labels
for i in range(1,8):
    paths = glob(f'../data/RAF-DB_balanced/train/{i}/*')
    img_paths += paths  # Assign label i to all images in folder i
img_paths = sorted(img_paths)
for path in img_paths : 
    label = int(path.split('/')[-2])
    img_labels.append(label)
img_labels = np.array(img_labels)  # Convert to numpy array for easy indexing

def load_and_transform(path):
    img = Image.open(path).convert('RGB')
    return transform(img)


# Parallel load/transform on CPU (unchanged, assuming you have it)


# Batched inference on GPU (unchanged)
if os.path.exists('checkpoint/emb_vectors.pt'):
    emb_vectors = torch.load('checkpoint/emb_vectors.pt')
else:
    emb_vectors = []
    with torch.no_grad():
        for i in tqdm(img_paths):
            img = load_and_transform(i)
            img = img.unsqueeze(0).to(device)
            outputs = model(img)
            # Assume first element of outputs are embeddings of shape (B, D)
            emb_vectors.append(outputs[0])
    # n_samples, feature_dim 
    emb_vectors = torch.nn.functional.normalize(torch.cat(emb_vectors, dim=0),dim=-1)
    print(emb_vectors.shape)
    # Normalize embeddings to unit length for cosine-similarity KMeans
    torch.save(emb_vectors, 'checkpoint/emb_vectors.pt')


# Modified function to build pairs with mutual NN, preferring different labels
def build_pairs_mutual_nn(E: torch.Tensor, img_labels: np.ndarray, chunk_size: int = 1024, topk: int = 10) -> np.ndarray:
    # E: [N, D], L2-normalized, on CPU or CUDA
    N = E.shape[0]
    device_local = E.device
    # ensure on CUDA in fp16 for memory efficiency
    E = E.to(device_local, dtype=torch.float16)
    nn_idx = torch.empty(N, dtype=torch.long, device='cpu')


    with torch.no_grad():
        for start in tqdm(range(0, N, chunk_size), desc="NN search", leave=False):
            end = min(start + chunk_size, N)
            chunk = E[start:end]  # [C, D]
            # cosine sim since E is normalized
            sim = torch.matmul(chunk, E.t())  # [C, N]
            # Get top-k to find a neighbor with different label
            topk_indices = torch.topk(sim, k=topk, dim=1).indices  # [C, topk]
            global_rows = torch.arange(start, end, device=sim.device)
            
            best = []
            labels_tensor = torch.tensor(img_labels, device=sim.device)
            
            for i in range(end - start):
                idx = global_rows[i].item()
                found = False
                for neighbor_idx in topk_indices[i].tolist():
                    if neighbor_idx != idx and labels_tensor[neighbor_idx] != labels_tensor[idx]:
                        best.append(neighbor_idx)
                        found = True
                        break
                if not found:
                    # Fallback to closest (even same label, excluding self)
                    for neighbor_idx in topk_indices[i].tolist():
                        if neighbor_idx != idx:
                            best.append(neighbor_idx)
                            break
            
            nn_idx[start:end] = torch.tensor(best, dtype=torch.long, device='cpu')
            del sim, topk_indices, chunk
            torch.cuda.empty_cache()


    nn_idx_np = nn_idx.numpy()
    visited = np.zeros(N, dtype=bool)
    labels_local = np.full(N, -1, dtype=np.int32)
    cluster_id = 0


    # mutual nearest neighbor pairing, only if labels differ
    for i in range(N):
        if visited[i]:
            continue
        j = int(nn_idx_np[i])
        if not visited[j] and int(nn_idx_np[j]) == i and img_labels[j] != img_labels[i]:
            labels_local[i] = cluster_id
            labels_local[j] = cluster_id
            visited[i] = True
            visited[j] = True
            cluster_id += 1


    # greedy pairing for leftovers (allow same label here to handle odds)
    remaining = np.where(~visited)[0]
    for k in range(0, len(remaining) - 1, 2):
        i = remaining[k]
        j = remaining[k + 1]
        labels_local[i] = cluster_id
        labels_local[j] = cluster_id
        cluster_id += 1


    return labels_local


labels = build_pairs_mutual_nn(emb_vectors.cuda(), img_labels)


# Select a random cluster (unchanged)
random_cluster = random.choice(range(4000))
cluster_indices = np.where(labels == random_cluster)[0]


# Display images from the selected cluster
# Create output directory and copy images instead of showing
out_dir = f"cluster_{random_cluster}_samples"
os.makedirs(out_dir, exist_ok=True)


# Copy up to 100 images from the selected cluster
for i, idx in enumerate(cluster_indices[:100]):
    src = img_paths[idx]
    base = os.path.basename(src)
    dst = os.path.join(out_dir, f"{i:03d}_" + base)
    try:
        shutil.copy2(src, dst)
    except Exception as e:
        print(f"Failed to copy {src}: {e}")


print(f"Copied {min(len(cluster_indices), 100)} images to {out_dir}")
