from torchvision import transforms
from torch.utils.data import Dataset
from glob import glob
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
import sys
sys.path.extend('..')
from utils import init_quality_model, by_ml
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import os
import torch.distributed as dist
import pickle
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from .sampler import ImbalancedDatasetSampler
from .sampler_wrapper import DistributedSamplerWrapper
import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import torch
from torch import distributed as dist
from PIL import Image
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import random

__all__ = ['FER','FER_KFOLD','ClassBatchSampler']

class FER(Dataset):
    def __init__(self,args,transform,train=True, idx=True, balanced=False):
        super().__init__()
        self.root = args.dataset_path
        self.train = train
        post = 'train' if train else ('test' if 'Affect' in self.root else ('valid_balanced' if balanced else 'valid' ))
        offset = 1 if 'RAF' in self.root else 0 
        self.transform=transform
        self.paths = []
        self.labels = []
        for i in range(7):
            paths = sorted(glob(f'{self.root}/{post}/{i+offset}/*'))
            self.paths += paths 
            self.labels += [i]*len(paths)
        self.labels = np.array(self.labels,dtype=np.int64)
        self.img_num_list = None 
        self.get_img_num_per_cls()
        self.idx = idx 
        # Optional in-memory preload of images for faster access
        self.pin_memory = getattr(args, 'pin_memory', False)
        self.path_to_index = {p: i for i, p in enumerate(self.paths)}
        self.preloaded_images = None
        if self.pin_memory:
            self.preloaded_images = []
            for p in tqdm(self.paths, disable=False):
                # Load image content into memory and detach from file handle
                img = Image.open(p)
                img = img.copy()
                self.preloaded_images.append(img)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self,idx):
        if self.preloaded_images is not None:
            img = self.preloaded_images[idx]
        else:
            img = Image.open(self.paths[idx])
        if isinstance(self.transform, list):
            samples = [tr(img) for tr in self.transform]
            if self.idx : 
                return samples, self.labels[idx] , idx
            else : 
                return samples, self.labels[idx] # len transforms, img 
        else:
            if self.idx : 
                return self.transform(img), self.labels[idx] , idx
            else : 
                return self.transform(img), self.labels[idx]
    
    def get_img_num_per_cls(self):
        if self.img_num_list is None:
            counter = Counter(self.labels)
            self.img_num_list = [0]*(np.max(self.labels)+1)
            for key, value in counter.items():
                self.img_num_list[key] = value
        return np.array(self.img_num_list)

class FER_KFOLD(FER):
    def __init__(self, args, transform ,n_folds=5, fold_idx=0, train=True, random_seed=42):
        super().__init__(args, transform, train=True)  # Always load all data, split later

        self.n_folds = n_folds
        self.fold_idx = fold_idx
        self.random_seed = random_seed
        self.train = train
        self.labels = np.array(self.labels)
        
        # self.paths: numpy array of all image paths, shape = (num_samples,)
        self.paths = np.array(self.paths)
        # self.original_paths: deep copy of all image paths, shape = (num_samples,)
        self.original_paths = deepcopy(self.paths)
        # self.indices: list of (train_idx, val_idx) tuples for each fold, each idx is a numpy array of indices
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        self.indices = list(kf.split(X=self.paths, y=self.labels))
        # train_idx: indices for training samples in this fold, shape = (num_train_samples,)
        # val_idx: indices for validation samples in this fold, shape = (num_val_samples,)
        train_idx, val_idx = self.indices[fold_idx]
        
        self.val_idx = val_idx
        if train:
            selected_idx = train_idx
        else:
            selected_idx = val_idx
        self.paths = self.paths[selected_idx]
        self.labels = self.labels[selected_idx]
        # Convert back to torch tensor for labels
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.paths)


    def get_indice_db(self):
        result = torch.cat((torch.tensor(self.val_idx),torch.tensor([self.fold_idx]*len(self.val_idx)))) #deterministic. 
        return result

class FER_uni(FER):
    def __init__(self,args, transform, train, idx, c):
        super().__init__(args, transform, train, idx)
        indices = self.labels[self.labels == c]
        self.paths = self.paths[indices.tolist()]
        self.labels = self.labels[indices]
    
    def __len__(self):
        return len(self.labels)


class ClassBatchSampler(FER):
    def __init__(self, args, transform, train=True, idx=True, balanced=False,
                 seed=42, num_workers=0, image_size=(112,112)):
        super().__init__(args, transform, train=train, idx=idx, balanced=balanced)
        # labels, class_to_idx 설정 (기존 스타일 유지)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        num_classes = int(torch.max(self.labels).item()) + 1 if len(self.labels) > 0 else 0
        self.class_to_idx = [[] for _ in range(num_classes)]
        for i, c in enumerate(self.labels.tolist()):
            self.class_to_idx[c].append(i)

        # DDP 환경 자동 감지
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        # 상태
        self.seed = int(seed)
        self.epoch = 0
        self.rng = random.Random(self._seed_for_epoch(self.epoch))
        self.num_workers = int(num_workers)
        self.image_size = image_size
        self.replace_if_insufficient = True
        # 전처리/로드 관련
        self._has_preloaded = getattr(self, 'preloaded_images', None) is not None
        self._path_to_index = getattr(self, 'path_to_index', None)

        # 클래스별 deque 구성
        self._build_deques()

    def _seed_for_epoch(self, epoch):
        return (self.seed + epoch * 1000003) & 0x7FFFFFFF

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        self.rng = random.Random(self._seed_for_epoch(self.epoch))
        self._build_deques()

    def _build_deques(self):
        self.deques = []
        for c, idxs in enumerate(self.class_to_idx):
            if len(idxs) == 0:
                self.deques.append(deque())
            else:
                shuffled = list(idxs)
                self.rng.shuffle(shuffled)
                self.deques.append(deque(shuffled))

    def _refill_if_needed(self, c: int):
        idxs = self.class_to_idx[c]
        if len(idxs) == 0:
            return
        shuffled = list(idxs)
        self.rng.shuffle(shuffled)
        self.deques[c].extend(shuffled)

    def _open(self, path):
        if self._has_preloaded and self._path_to_index is not None:
            i = self._path_to_index[path]
            img = self.preloaded_images[i]
            return img
        img = Image.open(path)
        return img

    def _apply_transform(self, img):
        # transform은 __init__에서 받은 것을 사용 (PIL augment → ToTensor → Normalize 권장)
        return self.transform(img) if self.transform is not None else img

    def _load_one(self, path):
        img = self._open(path)
        img = self._apply_transform(img)
        return img

    def __len__(self):
        return len(self.labels)

    def _sample_pairs_global(self, labels_batch: torch.Tensor, k: int):
        # 전역 무복원: 클래스 deque에서 pop으로 소비
        pairs = []  # (idx, class)
        for c in labels_batch.tolist():
            c = int(c)
            dq = self.deques[c]
            needed = k
            while needed > 0:
                if len(dq) == 0:
                    if self.replace_if_insufficient:
                        self._refill_if_needed(c)
                        dq = self.deques[c]
                        if len(dq) == 0:
                            raise ValueError(f"Class {c} empty; cannot refill.")
                    else:
                        raise ValueError(f"Class {c} insufficient for k={k}.")
                pairs.append((dq.popleft(), c))
                needed -= 1
        return pairs  # 길이 = bs*k

    def _shard_for_ddp(self, pairs):
        if self.world_size == 1:
            return pairs
        n = len(pairs)
        per_rank = (n + self.world_size - 1) // self.world_size  # ceil
        start = self.rank * per_rank
        end = min(start + per_rank, n)
        return pairs[start:end] if start < n else []

    def sample_pairs(self, labels_batch: torch.Tensor, k: int):
        """
        labels_batch: (bs,)
        return: images (bs_local, k, 3, 112, 112), labels (bs_local, k)
        """
        # 1) 전역 페어 생성
        pairs_global = self._sample_pairs_global(labels_batch, k)  # 무복원
        # 2) DDP shard
        pairs_local = self._shard_for_ddp(pairs_global)
        # 3) 로드 목록
        idxs = [i for i, _ in pairs_local]
        paths = [self.paths[i] for i in idxs]
        # 4) 멀티스레드 로딩+변환
        if self.num_workers > 0:
            with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
                imgs = list(ex.map(self._load_one, paths))
        else:
            imgs = [self._load_one(p) for p in paths]
        # 5) 텐서화/검증
        # transform이 Tensor를 반환하도록 구성되어 있어야 함
        if not all(isinstance(t, torch.Tensor) for t in imgs):
            raise TypeError("Transform must return torch.Tensor (e.g., ToTensor/Normalize 적용 필요).")
        batch = torch.stack(imgs, dim=0)  # (local_bs*k, C, H, W)
        # 크기 확인
        if batch.ndim != 4:
            raise ValueError(f"Expected 4D tensor, got {batch.shape}.")
        C, H, W = batch.shape[1:]
        if C != 3 or H != 112 or W != 112:
            raise ValueError(f"Expected (3,112,112), got {(C,H,W)}.")
        # 6) 라벨 정리 및 reshape
        y = torch.tensor([c for _, c in pairs_local], device=labels_batch.device, dtype=torch.long)
        total = batch.shape[0]
        if total % k != 0:
            # world_size 배수 정렬을 위해 마지막 잘라내기
            drop = total % k
            batch = batch[:-drop]
            y = y[:-drop]
            total = batch.shape[0]
        bs_local = total // k
        batch = batch.view(bs_local, k, C, H, W)
        y = y.view(bs_local, k)
        return batch.to(labels_batch.device)