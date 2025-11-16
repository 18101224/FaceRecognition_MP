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


__all__ = ['FER','FER_KFOLD','ClassBatchSampler']
class FER(Dataset):
    def __init__(self,args,transform,train=True, idx=True, balanced=False, imb_factor=1.0, debug=False):
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
        self.get_macro_category()
    def get_macro_category(self,):
        boundaries = {
            'AffectNet': [100000, 40000],
            'RAF-DB': [3000, 1500],
            'CAER': [3000, 1000]
        }

        # Infer dataset key from root path
        if 'Affect' in str(self.root):
            ds_key = 'AffectNet'
        elif 'RAF' in str(self.root):
            ds_key = 'RAF-DB'
        elif 'CAER' in str(self.root):
            ds_key = 'CAER'
        else:
            ds_key = None
        self.boundaries = boundaries[ds_key]
        counts = self.get_img_num_per_cls().astype(int)

        # Determine thresholds: use predefined if known dataset, otherwise use 67/33 quantiles
        if ds_key in boundaries:
            high, low = boundaries[ds_key]
        else:
            if len(counts) == 0:
                return {'many': [], 'medium': [], 'few': []}
            q67 = int(np.quantile(counts, 2/3))
            q33 = int(np.quantile(counts, 1/3))
            high, low = q67, q33

        categories = {'many': [], 'medium': [], 'few': []}
        for cls_idx, n in enumerate(counts):
            if n >= high:
                categories['many'].append(cls_idx)
            elif n <= low:
                categories['few'].append(cls_idx)
            else:
                categories['medium'].append(cls_idx)

        return categories
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
                return samples, self.labels[idx]
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

class ClassBatchSampler(FER) : 
    def __init__(self,args,transform,train=True, idx=True, balanced=False):
        super().__init__(args,transform,train=train, idx=idx, balanced=balanced)

        self.labels = torch.tensor(self.labels,dtype=torch.long)
        num_classes = torch.max(self.labels) + 1
        self.class_to_idx = [[] for _ in range(num_classes)]
        for i,c in enumerate(self.labels.tolist()):
            self.class_to_idx[c].append(i)

        self.class_counts = [len(v) for v in self.class_to_idx]
    
    def _load_one(self,path):
        if getattr(self, 'preloaded_images', None) is not None:
            i = self.path_to_index[path]
            img = self.preloaded_images[i]
        else:
            img = Image.open(path)
        if isinstance(self.transform, list):
            samples = [tr(img) for tr in self.transform]
            return samples
        else:
            return self.transform(img)
    
    def __len__(self):
        return len(self.labels)
    
    def sample_pairs(self,labels_batch,k=2,num_workers=0, replace_if_insufficient=True):
        '''
        labels_batch: bs
        return: (bs, k, 3, h, w), (bs, k)
        '''
    
        bs = labels_batch.numel()
        fetch_list = []
        for c in labels_batch.tolist():
            pool = self.class_to_idx[c]
            cnt = len(pool)
            if cnt >= k:
                perm = torch.randperm(cnt)[:k].tolist()
                picks = [pool[i] for i in perm]
            else:
                raise ValueError(f"Class {c} has only {cnt} samples, but {k} are required")
            fetch_list.extend( [self.paths[i] for i in picks] )
        
        if num_workers and num_workers > 0 :
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                imgs = list(executor.map(lambda p_c:self._load_one(p_c),fetch_list))
        else:
            imgs = [self._load_one(p_c) for p_c in fetch_list]

        # Handle case where _load_one returns a list (when transform is a list)
        if isinstance(self.transform, list) and len(imgs) > 0 and isinstance(imgs[0], list):
            # If transform is a list, imgs will be a list of lists
            # We need to flatten or select one transformation
            # For now, let's select the first transformation from each
            imgs = [img[0] if isinstance(img, list) else img for img in imgs]

        batch = torch.stack(imgs)
        batch = batch.view(bs,k,*batch.shape[1:])
        if batch.shape[1] == 1:
            batch = batch.squeeze(1)
            return batch.to(labels_batch.device), labels_batch.reshape(-1,1)
        return batch.to(labels_batch.device), labels_batch.unsqueeze(1).repeat(1,k).to(labels_batch.device)

    

