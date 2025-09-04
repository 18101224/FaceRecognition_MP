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


class FER(Dataset):
    def __init__(self,args,transform,train=True, idx=True, balanced=False):
        super().__init__()
        self.root = args.dataset_path
        self.train = train
        post = 'train' if train else ('test' if 'Affect' in self.root else ('valid' if ))
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

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self,idx):
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


