import numpy as np
import torch 
from tqdm import tqdm
from sklearn.manifold import TSNE

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
def get_predictions(model,loader,aligner=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preds = []
    labels = []
    for img, label in tqdm(loader):
        img = img.to(device)
        label = label.to(device)
        if aligner:
            _,_,ldmk,_,_,_ = aligner(img)
            pred = model(img,keypoint=ldmk)
        else:
            pred = model(img)
        preds.append(pred.detach().cpu())
        labels.append(label.detach().cpu())
    confs = torch.cat(preds,dim=0)
    preds = torch.argmax(confs,dim=1).numpy()
    labels = torch.cat(labels,dim=0).numpy()
    return preds, labels, confs.numpy()

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