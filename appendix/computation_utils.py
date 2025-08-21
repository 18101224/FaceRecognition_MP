import torch 
from tqdm import tqdm 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

__all__ = ['get_predictions', 'calculate_accuracy', 'get_angle_matrix', 'make_confusion_matrix']

@torch.no_grad()
def get_predictions(model, dataset, criterion, device):
    '''
    get predictions and losses
    '''
    predictions, losses, labels = [], [], []
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    for img, label in tqdm(dataloader):
        img = img.to(device)
        label = label.to(device)
        pred = model(img)
        loss = criterion(pred, label)
        # Handle scalar loss by unsqueezing
        if loss.dim() == 0:
            loss = loss.unsqueeze(0)
        labels.append(label)
        predictions.append(pred)
        losses.append(loss)
    return (
        torch.cat(predictions, dim=0).cpu(),
        torch.cat(losses, dim=0).cpu(),
        torch.cat(labels, dim=0).cpu()
    )


def calculate_accuracy(predictions, labels, metric='micro'):
    """
    Calculate accuracy based on predictions and labels.
    
    Args:
        predictions (torch.Tensor): Model predictions of shape (N, C) where N is number of samples and C is number of classes
        labels (torch.Tensor): Ground truth labels of shape (N,)
        metric (str): Either 'micro' or 'macro' accuracy
        
    Returns:
        float: Calculated accuracy
    """
    if metric not in ['micro', 'macro']:
        raise ValueError("metric must be either 'micro' or 'macro'")
    
    pred_labels = predictions.argmax(dim=1)
    num_classes = predictions.shape[1]
    
    if metric == 'micro':
        # Micro accuracy: overall accuracy across all samples
        return (pred_labels == labels).float().mean().item()
    
    else:  # macro accuracy
        # Macro accuracy: average of per-class accuracies
        accs = torch.zeros(num_classes, device=predictions.device)
        counts = torch.zeros(num_classes, device=predictions.device)
        
        # Calculate accuracy for each class
        for i in range(num_classes):
            mask = labels == i
            if mask.sum() > 0:
                accs[i] = (pred_labels[mask] == labels[mask]).float().sum()
                counts[i] = mask.sum()
        
        # Calculate accuracy for each class
        accuracies = []
        for i in range(len(counts)):
            if counts[i] > 0:
                accuracies.append(accs[i].item() / counts[i])
            else:
                accuracies.append(0.0)
        return accuracies

def get_angle_matrix(kernel):
    sims = kernel.T@kernel
    rad = torch.arccos(sims)
    degree = (rad * 180) / np.pi
    return degree


def make_confusion_matrix(preds, labels):
    """
    Create confusion matrix from predictions and labels.
    
    Args:
        preds (torch.Tensor): Model predictions
        labels (torch.Tensor): Ground truth labels
        
    Returns:
        numpy.ndarray: Integer confusion matrix
    """
    num_classes = preds.shape[1]
    conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int32)
    preds = torch.max(preds, dim=1)[1]
    for p, l in zip(preds, labels):
        conf_matrix[l, p] += 1
    return conf_matrix.cpu().numpy()