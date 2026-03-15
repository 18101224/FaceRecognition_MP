from __future__ import annotations

from pathlib import Path

import torch

from .classifiers.base import BaseClassifier
from .classifiers.partial_fc.partial_fc import PartialFC_V2
from .classifiers.fc.fc import FC


def get_classifier(classifier: str = "partial_fc", sample_rate=1.0, margin_loss_fn=None, output_dim=512, num_classes=0, rank=0, world_size=1):
    if classifier == "partial_fc":
        return PartialFCClassifier.build(
            sample_rate=sample_rate,
            margin_loss_fn=margin_loss_fn,
            output_dim=output_dim,
            num_classes=num_classes,
            rank=rank,
            world_size=world_size,
        )
    elif classifier == "fc":
        return FCClassifier.build(
            margin_loss_fn=margin_loss_fn,
            output_dim=output_dim,
            num_classes=num_classes,
            rank=rank,
            world_size=world_size,
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier}")


class PartialFCClassifier(BaseClassifier):
    def __init__(self, classifier, rank, world_size):
        super(PartialFCClassifier, self).__init__()
        self.partial_fc = classifier
        self.rank = rank
        self.world_size = world_size
        self.apply_ddp = False

    @classmethod
    def build(cls, sample_rate, margin_loss_fn, output_dim, num_classes, rank, world_size):
        classifier = PartialFC_V2(
            rank=rank,
            world_size=world_size,
            margin_loss=margin_loss_fn,
            embedding_size=output_dim,
            num_classes=num_classes,
            sample_rate=sample_rate,
        )
        model = cls(classifier, rank, world_size)
        model.eval()
        return model

    def forward(self, local_embeddings, local_labels):
        return self.partial_fc(local_embeddings, local_labels)


class FCClassifier(BaseClassifier):
    def __init__(self, classifier, rank, world_size):
        super(FCClassifier, self).__init__()
        self.classifier = classifier
        self.rank = rank
        self.world_size = world_size
        self.apply_ddp = True

    @classmethod
    def build(cls, margin_loss_fn, output_dim, num_classes, rank, world_size):
        classifier = FC(
            margin_loss=margin_loss_fn,
            embedding_size=output_dim,
            num_classes=num_classes,
        )
        model = cls(classifier, rank, world_size)
        model.eval()
        return model

    def forward(self, local_embeddings, local_labels):
        return self.classifier(local_embeddings, local_labels)

    def load_state_dict_from_path(self, pretrained_model_path):
        save_dir = Path(pretrained_model_path).parent
        save_name = Path(pretrained_model_path).name
        rank0_name = f"{Path(save_name).stem}_rank0{Path(save_name).suffix}"

        candidates = [
            save_dir / rank0_name,
            save_dir / save_name,
        ]
        ranked_paths = sorted(save_dir.glob(f"{Path(save_name).stem}_rank*.pt"))
        candidates.extend(ranked_paths)

        target_path = None
        for path in candidates:
            if path.exists():
                target_path = path
                break
        if target_path is None:
            raise FileNotFoundError(f"No classifier checkpoint found under {save_dir}")

        state_dict = torch.load(str(target_path), map_location="cpu", weights_only=False)
        result = self.load_state_dict(state_dict, strict=False)
        print("classifier loading result", result)
