import numpy as np
import torch 
from tqdm import tqdm
from sklearn.manifold import TSNE
from collections import Counter
from typing import Tuple

def compute_confusion_matrix(preds: np.ndarray, labels: np.ndarray):
    """
    Compute the confusion matrix from predictions and true labels.
    Args:
        preds (np.ndarray): Array of predicted class indices.
        labels (np.ndarray): Array of true class indices.
    Returns:
        np.ndarray: Confusion matrix of shape (num_classes, num_classes)
    """
    assert preds.shape == labels.shape, "Predictions and labels must have the same shape."
    num_classes = max(preds.max(), labels.max()) + 1
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm

def normalize_confusion_matrix(cm: np.ndarray, labels: np.ndarray):
    """
    Normalize the confusion matrix by the distribution of true labels.
    Each row (true label) is divided by the number of occurrences of that label in labels,
    so the sum of each row is 1 (if the label exists in labels).
    Args:
        cm (np.ndarray): Confusion matrix of shape (num_classes, num_classes)
        labels (np.ndarray): Array of true class indices
    Returns:
        np.ndarray: Normalized confusion matrix
    """
    label_counts = np.bincount(labels, minlength=cm.shape[0])
    # Avoid division by zero
    norm_cm = cm.astype(np.float32)
    for i, count in enumerate(label_counts):
        if count > 0:
            norm_cm[i, :] /= count
    return norm_cm

@torch.inference_mode()
def get_predictions(model,loader,aligner=None, get_features=True):
    '''
    returns : preds, labels, confs as numpy array 
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preds = []
    labels = []
    features = []
    features_branch = []
    indices = []
    for img, label, idx in tqdm(loader):
        img = img.to(device)
        label = label.to(device)
        if aligner:
            _,_,ldmk,_,_,_ = aligner(img)
            pred, feat_branch, c, feat  = model(img,keypoint=ldmk, features=True, wo_branch=True)
        else:
            pred, feat_branch, c, feat  = model(img, features=True, wo_branch=True)
        preds.append(pred.detach().cpu())
        labels.append(label.detach().cpu())
        if get_features : 
            features.append(feat.detach().cpu())
            features_branch.append(feat_branch.detach().cpu())
            indices.append(idx.detach().cpu())
    confs = torch.cat(preds,dim=0)
    preds = torch.argmax(confs,dim=1).numpy()
    labels = torch.cat(labels,dim=0).numpy()
    features = torch.cat(features,dim=0).numpy()
    features_branch = torch.cat(features_branch,dim=0).numpy()
    indices = torch.cat(indices,dim=0).numpy()
    return preds, labels, confs.numpy(), features, features_branch, c.detach().cpu().numpy(), indices

def get_macro_accuracy(preds, labels):
    counts = np.bincount(labels)
    macro_accuracy = np.zeros(len(counts))
    for i in range(len(counts)):
        macro_accuracy[i] = (preds[labels==i] == i).sum() / counts[i]
    macro_accuracy = macro_accuracy.mean()
    return macro_accuracy

@torch.no_grad()
def get_features(model, loader, aligner=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = []
    labels = []
    for img, label in tqdm(loader):
        img = img.to(device)
        if aligner:
            _,ldmk,_,_,_,_ = aligner(img)
            feature = model(img,ldmk)
            features.append(feature.detach().cpu())
        else:
            backbone_feat, cls_feat, bcl_feat, center_feat, _  = model.analysis(img)
            features.append((backbone_feat.detach().cpu(), cls_feat.detach().cpu(), bcl_feat.detach().cpu(), center_feat.detach().cpu()))
        labels.append(label.detach().cpu())
    if aligner:
        features = torch.cat(features, dim=0)
        features = torch.nn.functional.normalize(features, p=2, dim=1)  # Normalize each feature vector to norm 1
        return features.numpy(), torch.cat(labels,dim=0).numpy()
    else:
        return features, torch.cat(labels,dim=0).numpy()
    
def process_features(features):
    """
    Normalize and organize features into a list of numpy arrays per feature type.
    Accepts multiple input structures:
    - List[Tuple[Tensor,...]]: per-batch tuples of feature tensors -> returns list per feature type
    - List[Tensor]: per-batch single feature tensors -> returns [concat_normalized]
    - Tensor or np.ndarray of shape (N, D): returns [normalized_numpy]
    """
    # Case: already a single (N, D) tensor/array
    if isinstance(features, torch.Tensor):
        normalized = torch.nn.functional.normalize(features, p=2, dim=1)
        return [normalized.detach().cpu().numpy()]
    if isinstance(features, np.ndarray):
        tensor = torch.from_numpy(features)
        normalized = torch.nn.functional.normalize(tensor, p=2, dim=1)
        return [normalized.detach().cpu().numpy()]

    # Case: list container
    if isinstance(features, list) and len(features) > 0:
        first_elem = features[0]
        # List of tuples: multiple feature branches
        if isinstance(first_elem, (tuple, list)):
            num_feature_types = len(first_elem)
            result = []
            for idx in range(num_feature_types):
                per_type_tensors = [batch_features[idx] for batch_features in features]
                concatenated = torch.cat(per_type_tensors, dim=0)
                normalized = torch.nn.functional.normalize(concatenated, p=2, dim=1)
                result.append(normalized.detach().cpu().numpy())
            return result
        # List of tensors/arrays: single feature branch accumulated per batch
        elif isinstance(first_elem, (torch.Tensor, np.ndarray)):
            per_batch_tensors = [torch.from_numpy(x) if isinstance(x, np.ndarray) else x for x in features]
            concatenated = torch.cat(per_batch_tensors, dim=0)
            normalized = torch.nn.functional.normalize(concatenated, p=2, dim=1)
            return [normalized.detach().cpu().numpy()]

    raise ValueError("Unsupported features structure for process_features")


def compute_angle_matrix(model):
    weight = model.get_kernel().detach().cpu()
    angles = torch.arccos(weight.T @ weight) * 180.0 / torch.pi
    return angles.numpy()

def compute_error_rate_per_class(preds, labels):
    """
    Compute the error rate for each class.
    Args:
        preds (array-like): Predicted class indices for all samples.
        labels (array-like): True class indices for all samples.
    Returns:
        np.ndarray: Error rate for each class (length = num_classes)
    """
    preds = np.array(preds)
    labels = np.array(labels)
    num_classes = max(labels.max(), preds.max()) + 1
    error_rates = np.zeros(num_classes, dtype=float)
    for cls in range(num_classes):
        cls_mask = (labels == cls)
        total = cls_mask.sum()
        if total == 0:
            error_rates[cls] = np.nan  # or 0.0, depending on your preference
        else:
            errors = (preds[cls_mask] != cls).sum()
            error_rates[cls] = errors / total
    return error_rates

def compute_accuracy_per_class(preds, labels):
    """
    Compute the accuracy for each class.
    Args:
        preds (array-like): Predicted class indices for all samples.
        labels (array-like): True class indices for all samples.
    Returns:
        np.ndarray: Accuracy for each class (length = num_classes)
    """
    preds = np.array(preds)
    labels = np.array(labels)
    num_classes = max(labels.max(), preds.max()) + 1
    accuracies = np.zeros(num_classes, dtype=float)
    for cls in range(num_classes):
        cls_mask = (labels == cls)
        total = cls_mask.sum()
        if total == 0:
            accuracies[cls] = np.nan
        else:
            correct = (preds[cls_mask] == cls).sum()
            accuracies[cls] = correct / total
    return accuracies

def get_tsne_features(features, output_dim=2, **tsne_kwargs):
    """
    Perform t-SNE dimensionality reduction on feature vectors.
    Args:
        features (np.ndarray or torch.Tensor): Input feature vectors of shape (n_samples, n_features).
        output_dim (int): Output dimension for t-SNE (default: 2).
        **tsne_kwargs: Additional keyword arguments for sklearn.manifold.TSNE.
    Returns:
        np.ndarray: t-SNE reduced features of shape (n_samples, output_dim)
    """
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    tsne = TSNE(n_components=output_dim, random_state=42, **tsne_kwargs)
    reduced = tsne.fit_transform(features)
    return reduced

@torch.no_grad()
def get_features_from_backbone(backbone, loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = []
    labels = []
    for img, label in tqdm(loader):
        img = img.to(device)
        feature = backbone(img)
        feature = torch.nn.functional.normalize(feature, p=2, dim=1)
        features.append(feature.detach().cpu())
        labels.append(label.detach().cpu())
    features = torch.cat(features, dim=0)
    return features.numpy(), torch.cat(labels,dim=0).numpy()

def get_centers(features, labels):
    n_c = labels.max()+1 
    centers = []
    for y in range(n_c):
        indices = labels==y 
        dist = features[indices]
        center = dist.mean(axis=0).reshape(-1)
        center = center/((center*center).sum()**0.5)
        center = center.reshape(1,-1)
        centers.append(center)
    centers = np.concatenate(centers, axis=0)
    return centers


def find_nearest_training_for_misclassified(
    training_features: np.ndarray,
    validation_features: np.ndarray,
    validation_logits: np.ndarray,
    validation_labels: np.ndarray,
    target_centers: np.ndarray,
    k: int = 1,
    use_faiss: bool = False,
    batch_size: int = 8192,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    오분류된 validation 샘플만 선별 → target_centers로 CE 손실 계산 → 손실 내림차순 정렬 →
    정렬 순서대로 각 샘플의 top-k 가까운 training 인덱스와 유클리드 거리를 반환.

    입력:
    - training_features: (N_train, D) float
    - validation_features: (N_val, D) float
    - validation_logits: (N_val, C) float  # 오분류 판정용
    - validation_labels: (N_val,) int
    - target_centers: (C, D) float         # 손실 계산용 분류기 가중치/타깃 센터
    - k: top-k 근접 이웃 수
    - use_faiss: True면 Faiss IndexFlatL2 사용(설치 필요)
    - batch_size: NumPy 경로에서 쿼리 배치 크기

    반환:
    - mis_val_idx_sorted: (M,) 손실 내림차순으로 정렬된 오분류된 validation 인덱스
    - nn_indices: (M, k) 각 오분류 샘플의 top-k training 인덱스
    - nn_dist: (M, k) 유클리드 거리
    """
    # 1) 오분류 샘플 선별
    val_pred = np.argmax(validation_logits, axis=1)
    mis_mask = val_pred != validation_labels
    mis_val_idx = np.where(mis_mask)[0]
    if mis_val_idx.size == 0:
        return mis_val_idx, np.empty((0, 0), dtype=int), np.empty((0, 0), dtype=float)

    # 2) target_centers로 CE 손실 계산(Log-Sum-Exp 트릭, 수치안정)
    #    logits = val_feats @ target_centers^T
    val_feats_mis = validation_features[mis_val_idx].astype(np.float64, copy=False)  # (M, D)
    centers = target_centers.astype(np.float64, copy=False)                          # (C, D)
    logits = val_feats_mis @ centers.T                                              # (M, C)

    # CE(x,y) = - z_y + logsumexp(z)
    # logsumexp(z) = m + log(sum(exp(z - m))), where m = max(z)
    m = np.max(logits, axis=1, keepdims=True)                                       # (M, 1)
    z_shift = logits - m
    lse = m[:, 0] + np.log(np.sum(np.exp(z_shift), axis=1))                         # (M,)
    y_mis = validation_labels[mis_val_idx]
    z_y = logits[np.arange(logits.shape[0]), y_mis]                                  # (M,)
    ce_loss = -z_y + lse                                                             # (M,)

    # 손실 내림차순 정렬
    order = np.argsort(-ce_loss)
    mis_val_idx_sorted = mis_val_idx[order]
    # 이 순서로 쿼리 행렬 구성
    Q = validation_features[mis_val_idx_sorted]                                      # (M, D)

    # 3) top-k 최근접 트레이닝 검색
    X = training_features
    n_train = X.shape[0]
    k_eff = int(min(max(k, 1), n_train))

    # Faiss 경로(가능하면 사용)
    if use_faiss:
        try:
            import faiss  # type: ignore
            d = X.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(X.astype(np.float32, copy=False))
            D2, I = index.search(Q.astype(np.float32, copy=False), k_eff)  # L2^2
            D = np.sqrt(np.maximum(D2, 0.0))
            return mis_val_idx_sorted, I.astype(int, copy=False), D.astype(float, copy=False)
        except Exception:
            # 실패 시 NumPy 경로로 폴백
            pass

    # NumPy 경로(배치 처리, 벡터화 L2)
    X = X.astype(np.float64, copy=False)
    Q = Q.astype(np.float64, copy=False)
    X_norm2 = np.sum(X * X, axis=1)  # (N_train,)
    M = Q.shape[0]
    nn_indices = np.empty((M, k_eff), dtype=np.int64)
    nn_dists = np.empty((M, k_eff), dtype=np.float64)

    for s in range(0, M, batch_size):
        e = min(s + batch_size, M)
        Qb = Q[s:e]                                                    # (B, D)
        Qb_norm2 = np.sum(Qb * Qb, axis=1)                             # (B,)
        D2 = Qb_norm2[:, None] + X_norm2[None, :] - 2.0 * (Qb @ X.T)   # (B, N_train)
        D2 = np.maximum(D2, 0.0)
        part = np.argpartition(D2, kth=k_eff - 1, axis=1)[:, :k_eff]   # (B, k)  [unsorted]
        row_idx = np.arange(D2.shape[0])[:, None]
        chosen_D2 = D2[row_idx, part]
        order_in_row = np.argsort(chosen_D2, axis=1)
        topk_idx = part[row_idx, order_in_row]
        topk_D2 = chosen_D2[row_idx, order_in_row]
        nn_indices[s:e] = topk_idx
        nn_dists[s:e] = np.sqrt(topk_D2)

    return mis_val_idx_sorted, nn_indices.astype(int, copy=False), nn_dists.astype(float, copy=False)