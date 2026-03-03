from typing import Callable, Optional, Sequence, Any
from collections import Counter
import inspect

import torch
from torch.utils.data import Sampler
import torchvision


class ImbalancedDatasetSampler(Sampler[int]):
    """
    Imbalanced dataset sampler with controllable re-weighting power.

    For a sample i with label y_i and class count n_{y_i}:
        weight(i) = 1 / (n_{y_i} ** power)

    power = 1.0  -> fully balanced (inverse frequency)
    power = 0.5  -> sqrt sampling
    power = 0.0  -> no reweighting (uniform over samples)

    Args:
        dataset: torch Dataset
        labels: optional full label list aligned with dataset indices (len == len(dataset))
        indices: optional subset indices to sample from
        num_samples: number of samples per epoch/iteration
        callback_get_label:
            - Either a function that returns a full label list: f(dataset) -> Sequence
            - Or a function that returns one label: f(dataset, index) -> label
        power: alpha in [0, +inf). Typical is [0,1]. Larger than 1 over-boosts tails.
        replacement: whether to sample with replacement (default True)
        generator: optional torch.Generator for reproducibility
        eps: small value to avoid divide-by-zero (paranoia)
    """

    def __init__(
        self,
        dataset,
        labels: Optional[Sequence[Any]] = None,
        indices: Optional[Sequence[int]] = None,
        num_samples: Optional[int] = None,
        callback_get_label: Optional[Callable] = None,
        power: float = 1.0,
        replacement: bool = True,
        generator: Optional[torch.Generator] = None,
        eps: float = 1e-12,
    ):
        if power < 0:
            raise ValueError(f"power must be >= 0, got {power}")

        self.dataset = dataset
        self.indices = list(range(len(dataset))) if indices is None else list(indices)
        self.num_samples = len(self.indices) if num_samples is None else int(num_samples)

        self.callback_get_label = callback_get_label
        self.power = float(power)
        self.replacement = bool(replacement)
        self.generator = generator
        self.eps = float(eps)

        # Get labels for the FULL dataset (aligned by dataset index)
        full_labels = self._get_labels(dataset) if labels is None else labels
        if len(full_labels) != len(dataset):
            raise ValueError(
                f"labels must be aligned with dataset indices: "
                f"len(labels)={len(full_labels)} vs len(dataset)={len(dataset)}"
            )

        # Labels restricted to the sampler's index set
        labels_at_indices = [full_labels[i] for i in self.indices]

        # Count per class within the sampler's index set
        class_counts = Counter(labels_at_indices)

        # Per-sample weights: 1 / (count^power)
        # torch.multinomial accepts unnormalized positive weights.
        weights = []
        for y in labels_at_indices:
            c = class_counts[y]
            w = 1.0 / ((float(c) + self.eps) ** self.power)
            weights.append(w)

        self.weights = torch.as_tensor(weights, dtype=torch.double)

    def _get_labels(self, dataset) -> Sequence[Any]:
        # 1) User-provided callback
        if self.callback_get_label is not None:
            sig = inspect.signature(self.callback_get_label)
            # If callback expects (dataset, index) -> label, do per-index.
            if len(sig.parameters) >= 2:
                return [self.callback_get_label(dataset, i) for i in range(len(dataset))]
            # Else assume callback returns full label list: f(dataset) -> labels
            labels = self.callback_get_label(dataset)
            return list(labels)

        # 2) Common dataset types
        if isinstance(dataset, torch.utils.data.TensorDataset):
            # assume tensors[1] is labels
            y = dataset.tensors[1]
            return y.tolist() if hasattr(y, "tolist") else list(y)

        if isinstance(dataset, torchvision.datasets.MNIST):
            # newer torchvision uses dataset.targets; train_labels is legacy
            targets = getattr(dataset, "targets", None)
            if targets is None:
                targets = dataset.train_labels
            return targets.tolist()

        if isinstance(dataset, torchvision.datasets.ImageFolder):
            return [y for _, y in dataset.samples]

        if isinstance(dataset, torchvision.datasets.DatasetFolder):
            return [y for _, y in dataset.samples]

        if isinstance(dataset, torch.utils.data.Subset):
            # Need labels from the underlying dataset, then index by subset indices
            base = dataset.dataset
            base_labels = self._get_labels(base)
            return [base_labels[i] for i in dataset.indices]

        if isinstance(dataset, torch.utils.data.Dataset) and hasattr(dataset, "get_labels"):
            return list(dataset.get_labels())

        raise NotImplementedError(
            "Cannot infer labels. Provide `labels=` or `callback_get_label=` "
            "(either f(dataset)->labels or f(dataset,index)->label)."
        )

    def __iter__(self):
        sampled = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=self.replacement,
            generator=self.generator,
        )
        # sampled are positions in [0, len(self.indices))
        return (self.indices[i] for i in sampled.tolist())

    def __len__(self):
        return self.num_samples 