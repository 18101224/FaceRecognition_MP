import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

def cb_loss(preds: torch.Tensor,
            labels: torch.Tensor,
            samples_per_cls: torch.Tensor,
            beta: float,
            gamma: float = None,
            loss_type: str = 'softmax') -> torch.Tensor:
    """
    Compute Class-Balanced Loss (Cui et al.) with optional focal adjustment.

    Args:
        preds: Unnormalized logits, shape [batch_size, num_classes].
        labels: Ground-truth labels (int64), shape [batch_size].
        samples_per_cls: 1D tensor or list of length num_classes with class frequencies.
        beta: Scalar in [0,1) for effective number.
        gamma: Focusing parameter for focal loss; required if loss_type is 'focal'.
        loss_type: 'softmax', 'sigmoid', or 'focal'.

    Returns:
        Scalar loss tensor.
    """
    device = preds.device
    # Prepare class counts
    samples = torch.tensor(samples_per_cls, dtype=torch.float32, device=device)
    num_classes = preds.shape[1]

    # Compute weights based on effective number
    effective_num = 1.0 - beta ** samples
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * num_classes

    # Select per-sample weights
    sample_weights = weights[labels]

    if loss_type == 'softmax':
        ce = F.cross_entropy(preds, labels, reduction='none')
        loss = sample_weights * ce

    elif loss_type == 'sigmoid':
        one_hot = F.one_hot(labels, num_classes).float().to(device)
        bce = F.binary_cross_entropy_with_logits(preds, one_hot, reduction='none')
        loss = (sample_weights.unsqueeze(1) * bce).sum(dim=1)

    elif loss_type == 'focal':
        if gamma is None:
            raise ValueError("gamma must be provided for focal loss")
        one_hot = F.one_hot(labels, num_classes).float().to(device)
        prob = torch.sigmoid(preds)
        pt = one_hot * prob + (1 - one_hot) * (1 - prob)
        focal_factor = (1 - pt) ** gamma
        bce = F.binary_cross_entropy_with_logits(preds, one_hot, reduction='none')
        loss = (sample_weights.unsqueeze(1) * focal_factor * bce).sum(dim=1)

    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    return loss.mean()



class BalSCL(nn.Module):
    def __init__(self, cls_num_list=None, temperature=0.1):
        super(BalSCL, self).__init__()
        self.temperature = temperature
        self.cls_num_list = cls_num_list

    def forward(self, centers1, features, targets, ):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        batch_size = features.shape[0]
        targets = targets.contiguous().view(-1, 1).to(device)
        targets_centers = torch.arange(len(self.cls_num_list), device=device).view(-1, 1)
        targets = torch.cat([targets.repeat(2, 1), targets_centers], dim=0)
        batch_cls_count = torch.eye(len(self.cls_num_list), device=device)[targets].sum(dim=0).squeeze()

        mask = torch.eq(targets[:2 * batch_size], targets.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # class-complement
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        features = torch.cat([features, centers1], dim=0)
        logits = features[:2 * batch_size].mm(features.T)
        logits = torch.div(logits, self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # class-averaging
        exp_logits = torch.exp(logits) * logits_mask
        per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets], device=device).view(1, -1).expand(
            2 * batch_size, 2 * batch_size + len(self.cls_num_list)) - mask
        exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)
        
        log_prob = logits - torch.log(exp_logits_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.view(2, batch_size).mean()
        return loss



def weight_scheduling(method, beta, epoch, n_epochs):
    if method == 'linear':
        return beta * min(1.0, epoch / n_epochs)
    elif method == 'cosine':
        # epoch/n_epochs: 0~1 구간
        if epoch >= n_epochs:
            return beta
        return beta * (1 - np.cos(np.pi * epoch / (2 * n_epochs)))
    elif method == 'sigmoid':
        if epoch >= n_epochs:
            return beta
        phase = 1.0 - epoch / n_epochs
        return beta * np.exp(-5 * phase * phase)
    elif method == 'piecewise':
        # 예시: 0~30%는 0.1, 30~60%는 0.5, 이후는 1.0 (직접 수정해라)
        ratio = epoch / n_epochs
        if ratio < 0.3:
            return beta * 0.1
        elif ratio < 0.6:
            return beta * 0.5
        else:
            return beta * 1.0
    else:
        raise ValueError("Unknown method: {}".format(method))