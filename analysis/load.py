import sys
sys.path.append('..')
from models import ImbalancedModel
from dataset import get_cifar_dataset, get_transform, Large_dataset
import torch
from torch.utils.data import DataLoader 
from aligners import get_aligner
import argparse

def get_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = args.dataset_name
    model_type = args.model_type
    if 'cifar' in dataset_name :
        n_c = 100 if '100' in dataset_name else 10
        model = ImbalancedModel(num_classes=n_c, model_type=model_type,feature_module=args.feature_module, feature_branch=args.feature_branch)
        model = model.to(device)
        model.eval()
        return model
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
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
    return result
    
def load_loaders(train_set, vaild_set):
    train_loader = DataLoader(train_set, batch_size=128, shuffle=False, num_workers=16, pin_memory=True, generator=torch.Generator().manual_seed(42))
    valid_loader = DataLoader(vaild_set, batch_size=128, shuffle=False, num_workers=16, pin_memory=True, generator=torch.Generator().manual_seed(42))
    return train_loader, valid_loader

def load_aligner(config_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_aligner(config_path)
    model = model.to(device)
    model.eval()
    return model

