import sys 
sys.path.append('..')
from dataset import FER, ImbalanceSVHN, CIFARLTHF
from models import CosClassifier, CIFAR_RepVGG, get_imgnet_resnet
import torch 


__all__ = ['load_model', 'load_dataset']


def load_dataset(args):
    if 'cifar' in args.dataset_name:
        n_classes = 100 if '100' in args.dataset_name else 10
        train_set = CIFARLTHF(n_classes, train=True,)
        valid_set = CIFARLTHF(n_classes, train=False)
    elif 'svhn' in args.dataset_name:
        train_set = ImbalanceSVHN('../data', split='train',download=True)
        valid_set = ImbalanceSVHN('../data', split='test',download=True)
    elif 'affectnet' in args.dataset_name:
        train_set = FER(args, 'checkpoint/quality', train=True)
        valid_set = FER(args, 'checkpoint/quality', train=False)
    else:
        raise ValueError(f'Dataset {args.dataset_name} not supported')
    return train_set, valid_set

def load_model(args,trained_ckpt, pretrained_ckpt=None):
    device = torch.device('cuda')
    if 'cifar' in args.dataset_name:
        n_classes = 100 if '100' in args.dataset_name else 10
        model = CIFAR_RepVGG(pretrained_path=pretrained_ckpt,num_classes=n_classes).to(device)
        model.load_state_dict(torch.load(trained_ckpt,map_location=device))
        model.eval()
    elif 'svhn' in args.dataset_name:
        bacbone = get_imgnet_resnet()
        model = CosClassifier(num_classes=10, backbone=bacbone, dim=2048, num_hidden_layers=4).to(device)
        model.load_state_dict(torch.load(trained_ckpt,map_location=device)['model_state_dict'])
        model.eval()
    elif 'affectnet' in args.dataset_name:
        model = CosClassifier(num_classes=7, backbone=bacbone, dim=2048, num_hidden_layers=4)
    return model

