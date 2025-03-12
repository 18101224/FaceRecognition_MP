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
from .kp_rpe import *
import numpy as np
from tqdm import tqdm
import os
import torch.distributed as dist
import pickle
from collections import defaultdict

def exc_label(i):
    # Affect to RAF label
    label = [6,3,4,0,1,2,5]
    return label[i]



class FER(Dataset):
    def __init__(self,args,ckpt_path,train=True):
        super().__init__()
        self.root = args.dataset_path
        self.train = train
        if train:
            post = 'train'
            self.transforms = transforms.Compose([
                transforms.Resize((112, 112)),
                transforms.RandomResizedCrop(112, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3)
            ])
            memfile_path = os.path.join(args.dataset_path, 'memfile_train')
        else:
            post = 'valid' if 'RAF' in self.root else 'test'
            self.transforms = transforms.Compose([
                transforms.Resize(112),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3)
            ])
            memfile_path = os.path.join(args.dataset_path,'memfile_valid')
        if os.path.exists(memfile_path):
            self.paths, self.labels, self.class_qualities, self.qualities = self.load_memfile()
        if args.world_size > 1 :
            dist.barrier()
        if not os.path.exists(memfile_path):
            if args.world_size == 1 or args.rank == 0:
                os.mkdir(memfile_path)

                model = init_quality_model(ckpt_path)
                paths = []
                labels = []
                qualities = []
                class_qualities = [0]*7
                for i in range(7):
                    dir = glob(f'{self.root}/{post}/{i}/*')
                    paths += dir
                    label = exc_label(i) if 'Affect' in self.root else i
                    labels += [label]*len(dir)
                    qs = []
                    for path in dir:
                        qs.append(by_ml(model,path))
                    qualities.append(qs)
                    class_qualities[label] = qs
                dataset = {
                    'img_paths':paths,
                    'labels':labels,
                    'class_qualities':class_qualities,
                    'qualities':qualities
                }
                with open(os.path.join(memfile_path,'mem.pkl'),'wb') as f:
                    pickle.dump(dataset,f)

            if args.world_size >1 :
                dist.barrier()

            self.paths, self.labels, self.class_qualities, self.qualities = self.load_memfile()

    def load_memfile(self):
        post = 'train' if self.train else 'valid'
        with open(f'{self.root}/memfile_{post}/mem.pkl','rb') as f:
            dataset = pickle.load(f)
        labels = torch.tensor(dataset['labels'],dtype=torch.long)
        qualities = []
        for qs in dataset['qualities']:
            qualities+=qs
        qualities = np.array(qualities)
        return_qualities = (qualities-qualities.mean())/qualities.std()
        class_qualities = [0]*7
        for label, qs in enumerate(dataset['class_qualities']):
            class_qualities[label] = sum(qs)/len(qs)
        class_qualities = torch.tensor(class_qualities)
        class_qualities = (class_qualities - qualities.mean())/qualities.std()
        class_qualities = nn.functional.softmax(-class_qualities)
        return dataset['img_paths'], labels, class_qualities, return_qualities

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self,idx):
        img = Image.open(self.paths[idx])
        img = self.transforms(img)
        return img, self.labels[idx], self.qualities[idx], self.class_qualities[self.labels[idx]]





class raf(Dataset):
    def __init__(self,root,ckpt_path,train=True):
        super().__init__()
        if train :
            post = 'train'
        else:
            post = 'valid'

        label_offset = 1 if 'RAF' in root else 0
        self.transforms = transforms.Compose([
            transforms.Resize(112),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3,[0.5]*3)
        ])
        model = init_quality_model(ckpt_path)
        root = f'{root}/{post}'
        self.paths = []
        self.labels = []
        self.qualities = []
        for i in tqdm(range(label_offset,7+label_offset),desc=f'measuring quality for {post}'):
            paths = glob(root+f'/{i}/*')
            self.paths+=paths
            self.labels+=[i-label_offset]*len(paths)
            qualities = []
            for path in paths:
                qualities.append(by_ml(model,path))
            self.qualities+=qualities

        self.qualities = np.array(self.qualities)
        self.qualities = self.qualities/self.qualities.mean()
        self.label = deepcopy(self.labels)
        self.labels = torch.tensor(self.labels,dtype=torch.long)
        self.len = len(self.labels)

    def __getitem__(self,idx):
        img = Image.open(self.paths[idx])
        img = self.transforms(img)
        return img, self.labels[idx], self.qualities[idx]

    def __len__(self):
        return self.len

    def get_label(self):
        return self.label

class AffectNet(raf):
    def __init__(self,root,ckpt_path,train=True):
        super().__init__(root,ckpt_path,train)
        if train :
            post = 'train'
        else:
            post = 'valid' if 'RAF' not in root else 'test'
        root = f'{root}/{post}'
        self.labels = []
        for i in range(7):
            paths = glob(root+f'/{i}/*')
            label = exc_label(i)
            self.labels += [label]*len(paths)
        self.labels = torch.tensor(self.labels,dtype=torch.long)


class selected(raf):
    def __init__(self,root, indices):
        super().__init__(root, train=True)
        self.paths = [self.paths[i] for i in indices]
        self.labels = torch.tensor([self.labels[i] for i in indices], dtype=torch.long)
        self.len = len(indices)
    def __getitem__(self,idx):
        img = Image.open(self.paths[idx])
        img = self.transforms(img)
        return img, self.labels[idx], 0

    def __len__(self):
        return self.len

class raf_for_kp(raf):
    def __init__(self,root,ckpt_path,train=True):
        super().__init__(root,ckpt_path,train)
        if train:
            self.transforms = get_kprpe_transform_train()
        else:
            self.transforms = get_kprpe_transform_valid()
class AffectNet_for_kp(AffectNet):
    def __init__(self,root,ckpt_path,train=True):
        super().__init__(root,ckpt_path,train)
        if train:
            self.transforms = get_kprpe_transform_train()
        else:
            self.transforms = get_kprpe_transform_valid()
