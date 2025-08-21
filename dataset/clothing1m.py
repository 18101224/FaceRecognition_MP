from torch.utils.data import Dataset
import numpy as np 
import os 
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy
import torch 
from glob import glob

class Clothing1m(Dataset):
    def __init__(self, args, transform, train=True):
        super().__init__()
        root = args.dataset_path
        postfix = 'train' if train else 'valid'
        self.paths = []
        self.labels = []
        for i in range(14):
            paths = sorted(glob(f'{root}/{postfix}/{i}/*'))
            self.paths += paths 
            self.labels += [i]*len(paths)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.transform = transform

    def __getitem__(self,idx):
        img = Image.open(self.paths[idx])
        if self.transform is not None :
            img = self.transform(img)
        return img, self.labels[idx], idx 
    
    def __len__(self):
        return len(self.labels)
    
class Clothing1m_KFOLD(Clothing1m):
    def __init__(self, args, n_folds:int, fold_idx:int ,transform=None, train=True,random_seed=42):
        super().__init__(args, train=True, transform=transform)
        print(f'n_folds : {n_folds}, fold_idx : {fold_idx}, random_seed : {random_seed}')
        self.paths = np.array(self.paths)
        self.n_folds = n_folds 
        self.fold_idx = fold_idx 
        self.random_seed = random_seed 
        self.train = train 
        self.original_paths = deepcopy(self.paths)
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed,)
        self.indices = list(kf.split(X=self.paths, y=self.labels))
        train_idx, val_idx = self.indices[fold_idx]
        self.val_idx = val_idx 
        selected_idx = train_idx if train else val_idx
        selected_idx = torch.tensor(selected_idx).to(torch.long)
        self.paths = self.paths[selected_idx]
        self.labels = self.labels[selected_idx]
    
    def get_indice_db(self):
        result = torch.cat((torch.tensor(self.val_idx),torch.tensor([self.fold_idx]*len(self.val_idx)))) #deterministic. 
        return result
    
