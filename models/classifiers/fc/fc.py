from typing import Callable

import torch
from torch.nn.functional import linear, normalize

from losses.adaface import AdaFaceLoss



class FC(torch.nn.Module):
    def __init__(
        self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
    ):
        super(FC, self).__init__()

        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (self.num_classes, embedding_size)))

        if callable(margin_loss):
            self.margin_softmax = margin_loss
        else:
            raise TypeError("`margin_loss` must be callable.")

    def forward(
        self,
        local_embeddings: torch.Tensor,
        local_labels: torch.Tensor,
    ):
        embeddings = local_embeddings
        labels = local_labels.reshape(-1).long()

        norms = embeddings.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
        norm_embeddings = embeddings / norms

        norm_weight_activated = normalize(self.weight)
        logits = linear(norm_embeddings, norm_weight_activated)
        logits = logits.clamp(-1, 1)

        if not isinstance(self.margin_softmax, AdaFaceLoss):
            raise ValueError("FC classifier currently supports AdaFaceLoss only.")

        logits = self.margin_softmax(logits=logits, labels=labels, norms=norms)

        loss = self.cross_entropy(logits, labels)
        return loss
