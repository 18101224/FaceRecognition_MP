from torchvision import transforms 
import torch
import random
import numpy as np
from .augmentations import ImageNetPolicy, CIFAR10Policy, Cutout, GaussianBlur
from PIL import ImageFilter

__all__ = ['get_transform', 'get_imagenet_transforms']

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)
imgnet_mean = (0.485, 0.456, 0.406)
imgnet_std = (0.229, 0.224, 0.225)
inat_mean = (0.466, 0.471, 0.380)
inat_std = (0.195, 0.194, 0.192)
fer_mean = [ [0.5] * 3, imgnet_mean]
fer_std = [ [0.5] * 3, imgnet_std]
def include(loss, loss_types):
    losses = loss.split('_')
    for loss_type in loss_types:
        for loss in losses :
            if loss == loss_type :
                return True 
    return False 

def get_transform(args, train):
    dataset_name = args.dataset_name
    dataset_name = 'cifar' if 'cifar' in dataset_name else dataset_name
    if not train : 
        validation_transforms = {
            'cifar' : transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
            ]),
            'imagenet_lt' : transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=imgnet_mean, std=imgnet_std)
            ]),
            'RAF-DB' : transforms.Compose([
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(fer_mean[int(args.model_type!='kprpe')], fer_std[int(args.model_type!='kprpe')])
            ])
        }
        validation_transforms = {
            'AffectNet' : validation_transforms['RAF-DB'],
            **validation_transforms
        }
        return validation_transforms[dataset_name]

    
    if include(args.loss, ['CE', 'LDAM', 'BS', 'DRW', 'RIDE']):
        large_size_tr = [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4,0.4,0.4,0.1),
                ImageNetPolicy(),
                transforms.ToTensor(),]
        tr_dict= {
        'cifar': [transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(), # AutoAug
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize(
                mean=cifar10_mean, std=cifar10_std
            )
        ])]*3, 
        'imagenet_lt': [
            transforms.Compose([
                transforms.RandomResizedCrop(224),*large_size_tr,
                transforms.Normalize(mean=imgnet_mean, std=imgnet_std)
            ])
            ],
        'RAF-DB' : [
            transforms.Compose([
                transforms.Resize((112, 112)),
                *large_size_tr,
                transforms.Normalize(fer_mean[int(args.model_type!='kprpe')], fer_std[int(args.model_type!='kprpe')])
            ])
        ],
        'AffectNet' : [
            transforms.Compose([
                transforms.Resize((112, 112)),
                *large_size_tr,
                transforms.Normalize(fer_mean[int(args.model_type!='kprpe')], fer_std[int(args.model_type!='kprpe')])
            ])
        ]
        }
    elif include(args.loss, ['NCL']):
        tr_dict = {
            'cifar': [
            transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(), # AutoAug
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16),
                transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
            ]),
            transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(0.2,1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RadomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)]),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
            ])],
            'imagenet_lt': [
                transforms.Compose([
                  transforms.RandomResizedCrop(224,scale=(0.08,1.)),
                  transforms.RandomHorizontalFlip(),
                  transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.0)],p=1.),
                  ImageNetPolicy(),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=imgnet_mean, std=imgnet_std)
                ]),
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)],p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=imgnet_mean, std=imgnet_std)
                ])
            ]
        }
    elif include(args.loss, ['BCL']):
        large_before = [transforms.RandomHorizontalFlip(),
                  transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.0)],p=1.),
                  ImageNetPolicy(),
                  transforms.ToTensor()]
        large_after = [ transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)],p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),]
        tr_dict = {
            'cifar': [
                transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    CIFAR10Policy(), # AutoAug
                    transforms.ToTensor(),
                    Cutout(n_holes=1, length=16),
                    transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
                ]),
                *[transforms.Compose([
                    transforms.RandomResizedCrop(32, scale=(0.2,1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)],p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
                ])]*2 
            ],
            'imagenet_lt': [
                transforms.Compose([
                  transforms.RandomResizedCrop(224,scale=(0.08,1.)),
                  *large_before,
                  transforms.Normalize(mean=imgnet_mean, std=imgnet_std)
                ]),
                *[transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    *large_after,
                    transforms.Normalize(mean=imgnet_mean, std=imgnet_std)
                ])]*2
            ],
            'RAF-DB' : [
                transforms.Compose([
                    transforms.Resize((112, 112)),
                    *large_before,
                    transforms.Normalize(fer_mean[int(args.model_type!='kprpe')], fer_std[int(args.model_type!='kprpe')])
                ]),
                *[transforms.Compose([
                    transforms.RandomResizedCrop(112),
                    *large_after,
                    transforms.Normalize(fer_mean[int(args.model_type!='kprpe')], fer_std[int(args.model_type!='kprpe')])
                ])]*2
            ],
            'AffectNet' : [
                transforms.Compose([
                    transforms.Resize((112, 112)),
                    *large_before,
                    transforms.Normalize(fer_mean[int(args.model_type!='kprpe')], fer_std[int(args.model_type!='kprpe')])
                ]),
                *[transforms.Compose([
                    transforms.RandomResizedCrop(112),
                    *large_after,
                    transforms.Normalize(fer_mean[int(args.model_type!='kprpe')], fer_std[int(args.model_type!='kprpe')])
                ])]*2
            ]
        }
    else:
        raise ValueError(f"Loss {args.loss} not supported")
    if not args.aug :
        tr_dict[dataset_name][0].transforms.pop(2)

    return tr_dict[dataset_name]


def get_imagenet_transforms(args, train):
    return 

def get_clothing1m_transforms(train):
    mean, std = (0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)
    if train : 
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else : 
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

def get_fer_transforms(train):
    if train :
        return transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.RandomResizedCrop(112, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
    else : 
        return transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])