import sys
sys.path.append('..')
from dataset.fer import FER
from dataset.transform import get_transform
from models import  get_model 
from dataset import get_cifar_dataset, get_transform, Large_dataset, get_fer_transforms
import torch
from torch.utils.data import DataLoader 
from aligners import get_aligner
import argparse
import os 
from functools import partial

def load_logs(ckpt_paths):
    results = []
    for ckpt_path in ckpt_paths:
        results.append(
            torch.load(os.path.join(ckpt_path,'best.pth'),weights_only=False)
        )
    return results

def concat_args(args, logs):
    results = []
    temp_args = vars(args) if isinstance(args,argparse.Namespace) else args 
    for log in logs : 
        temp_log_args = vars(log['args']) if isinstance(log['args'],argparse.Namespace) else log['args']
        temp_log_args = {**temp_log_args, **log['model_params']}
        results.append(argparse.Namespace(**{**temp_args, **temp_log_args}))
    return results

def load_models(model_paths, args):
    results = []
    for model_path, arg in zip(model_paths, args) : 
        ckpt_path = os.path.join(model_path, f'{arg.ckpt_type}.pth')
        model = get_model(arg, ckpt_path)
        model.load_state_dict(torch.load(ckpt_path,map_location=torch.device('cuda'),weights_only=False)['model_state_dict'])
        model = model.to(torch.device('cuda'))
        model.eval()
        results.append(model)
    return results 

def load_model(model, ckpt_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(ckpt_path,map_location=device))
    model.eval()
    model.to(device)
    return model

def load_dataset(args,dataset_path, dataset_name, imb_factor=None):
    result = []

    for train in ['train', 'test']:
        if 'cifar' in dataset_name : 
            n_c = 100 if '100' in dataset_name else 10
            dataset =  get_cifar_dataset(dataset_name=dataset_name, root=dataset_path, imb_factor=0.01, imb_type='exp',train=(train=='train'), transform=get_transform(args,train=False))
            result.append(dataset)
        elif 'imagenet_lt' in dataset_name:
            dataset = Large_dataset(root=dataset_path, mode=train, transform=None)
            result.append(dataset)
        elif 'RAF-DB' in dataset_name:
            dataset = FER(args, train=(train=='train'), transform=get_fer_transforms(train=False, model_type=args.model_type), idx=True, debug=True)
            result.append(dataset)
            if not train == 'train':
                result.append(FER(args, train=(train=='train'), transform=get_fer_transforms(train=False, model_type=args.model_type), balanced=True, idx=True, debug=True))
        elif 'AffectNet' in dataset_name:
            result.append(FER(args, train=(train=='train'), transform=get_fer_transforms(train=False, model_type=args.model_type), idx=True, debug=True))
        elif 'CAER' in dataset_name:
            result.append(FER(args=args, train=(train=='train'), transform=get_fer_transforms(train=False, model_type=args.model_type), idx=True, imb_factor=args.imb_factor, debug=True))
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

