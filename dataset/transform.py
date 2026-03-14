from torchvision import transforms 
import torch
from copy import deepcopy
from timm.data.auto_augment import rand_augment_transform

__all__ = ['get_transform', 'get_imagenet_transforms', 'random_masking',
 'point_block_mask','get_fer_transforms', 'masking_pair', 'get_multi_view_transforms']

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

def isfer(dataset_name):
    if dataset_name in ['RAF-DB', 'AffectNet', 'CAER']:
        return True
    return False

def imagenet_multiview():
    global imgnet_mean, imgnet_std
    mean_t = torch.tensor(imgnet_mean)
    return rand_augment_transform(
        config_str='rand-m9-mstd0.5',
        hparams={
            'transrate_const': 224 * 0.5,
            'img_min' : (mean_t * 255).int().tolist() 
        }
    )

def get_transforms(train,model_type,args=None):
    if isfer(args.dataset_name):
        mean=[0.485, 0.456, 0.406] if 'kprpe' not in model_type else [0.5] * 3
        std=[0.229, 0.224, 0.225] if 'kprpe' not in model_type else [0.5] * 3
    else:
        mean = imgnet_mean
        std = imgnet_std
    if train :
        if isfer(args.dataset_name):
            pipeline = [
                transforms.Resize((112, 112)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05)
                ], p=0.2),                  # 밝기·대비·채도 너무 강하지 않게, 실무 validation 권고 적용
                transforms.RandomRotation(10, fill=(0,0,0)),  # 약한 회전
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3)
                ], p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                transforms.RandomErasing(p=0.25, scale=(0.02,0.33), value='random', inplace=False)
            ]
            # 필요 시 CutOut/Erase/Crop 증강 더 추가 가능 (landmark ROI 안전 검사 필요)
            # 예) custom cutout(눈·입 방어)는 albumentations나 직접 커스텀 함수로 구현
            return transforms.Compose(pipeline)
        else:
            ra = imagenet_multiview()
            pipeline = [
                transforms.RandomResizedCrop(224, 
                scale=(0.6,1.), ratio=(3/4, 4/3), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
            pipeline.append(ra)
            pipeline.extend([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
            return transforms.Compose(pipeline)
    else : 
        if isfer(args.dataset_name):
            return transforms.Compose([
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            return transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

def get_multi_view_transforms(args, train, model_type):
    mean=[0.485, 0.456, 0.406] if 'kprpe' not in model_type else [0.5] * 3
    std=[0.229, 0.224, 0.225] if 'kprpe' not in model_type else [0.5] * 3
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    if isfer(args.dataset_name):
        ra = rand_augment_transform(
            config_str='rand-m9-mstd0.5',
            hparams={
                'transrate_const': int(args.img_size*0.5),
                'img_min' : (mean * 255).int().tolist() 
            }
        )
        view_tf = transforms.Compose([
        transforms.Resize(args.img_size),
        ra, 
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])
    else:
        view_tf = get_transforms(train, model_type,args)

    if getattr(args, 'use_view', False) and train : 
        return [get_transforms(train, model_type,args), deepcopy(view_tf), deepcopy(view_tf)]
    else:
        return get_transforms(train, model_type,args)
