import sys
sys.path.append('..')
from dataset.fer import FER
from dataset.transform import get_transform
from models import  get_model 
from dataset import get_cifar_dataset, get_transform, Large_dataset
import torch
from torch.utils.data import DataLoader 
from aligners import get_aligner
import argparse


def load_model(model, ckpt_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(ckpt_path,map_location=device))
    model.eval()
    model.to(device)
    return model

def load_dataset(dataset_path, dataset_name, imb_factor=None):
    result = []
    for train in ['train', 'test']:
        if 'cifar' in dataset_name : 
            n_c = 100 if '100' in dataset_name else 10
            args = argparse.Namespace(**{'dataset_name':dataset_name})
            dataset =  get_cifar_dataset(dataset_name=dataset_name, root=dataset_path, imb_factor=0.01, imb_type='exp',train=(train=='train'), transform=get_transform(args,train=False))
            result.append(dataset)
        elif 'imagenet_lt' in dataset_name:
            dataset = Large_dataset(root=dataset_path, mode=train, transform=None)
            result.append(dataset)
        elif 'RAF-DB' in dataset_name:
            dataset = FER(args, get_transform(args,train=(train=='train')))
            result.append(dataset)
            if not train == 'train':
                result.append(FER(args, get_transform(args,train=False),balanced=True))
        elif 'AffectNet' in dataset_name:
            raise ValueError(f'Dataset {dataset_name} not supported')
                
    return result
    
def load_loaders(datasets):
    result = []
    for dataset in datasets:
        loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=16, pin_memory=True, generator=torch.Generator().manual_seed(42))
        result.append(loader)
    return result

def load_aligner(config_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_aligner(config_path)
    model = model.to(device)
    model.eval()
    return model

