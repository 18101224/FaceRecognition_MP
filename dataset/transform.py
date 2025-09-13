from torchvision import transforms 
import torch
import random
import numpy as np
from .augmentations import ImageNetPolicy, CIFAR10Policy, Cutout, GaussianBlur
from PIL import ImageFilter

__all__ = ['get_transform', 'get_imagenet_transforms', 'random_masking', 'point_block_mask']

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
                transforms.CenterCrop(224 if args.dataset_name != 'imagenet_lt_small' else 112),
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
        if 'imagenet_lt_small' == args.dataset_name : 
            dataset_name='imagenet_lt'
        return validation_transforms[dataset_name]

    loss = getattr(args,'loss','CE')
    if not include(loss, ['CE', 'LDAM', 'BS', 'DRW', 'RIDE', 'NCL', 'BCL']):
        loss = 'CE'
    if include(loss, ['CE', 'LDAM', 'BS', 'DRW', 'RIDE']):
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
                transforms.RandomResizedCrop(224 if args.dataset_name != 'imagenet_lt_small' else 112),*large_size_tr,
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
    elif include(loss, ['NCL']):
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
                  transforms.RandomResizedCrop(224 if args.dataset_name != 'imagenet_lt_small' else 112,scale=(0.08,1.)),
                  transforms.RandomHorizontalFlip(),
                  transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.0)],p=1.),
                  ImageNetPolicy(),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=imgnet_mean, std=imgnet_std)
                ]),
                transforms.Compose([
                    transforms.RandomResizedCrop(224 if args.dataset_name != 'imagenet_lt_small' else 112),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)],p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=imgnet_mean, std=imgnet_std)
                ])
            ]
        }
    elif include(loss, ['BCL']):
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
                  transforms.RandomResizedCrop(224 if args.dataset_name != 'imagenet_lt_small' else 112,scale=(0.08,1.)),
                  *large_before,
                  transforms.Normalize(mean=imgnet_mean, std=imgnet_std)
                ]),
                *[transforms.Compose([
                    transforms.RandomResizedCrop(224 if args.dataset_name != 'imagenet_lt_small' else 112),
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
        raise ValueError(f"Loss {loss} not supported")
    if not getattr(args, 'aug', False) :
        tr_dict[dataset_name][0].transforms.pop(2)
    if 'imagenet_lt_small' == args.dataset_name : 
        dataset_name='imagenet_lt'
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

def get_fer_transforms(train,):
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

def gaussian_prob(distance, sigma=1.5):
    return np.exp(-0.5*(distance/sigma)**2)


@torch.no_grad()
def random_masking(bs, img_size, block_size, n_blocks, device):
    """
    Generate a random block mask on a coarse grid.

    Args:
        bs (int): Batch size.
        img_size (int): Image height/width in pixels. Must be divisible by `block_size`.
        block_size (int): Side length of a square block in pixels.
        n_blocks (int): Number of blocks to set to 1 per image. Must satisfy
            0 <= n_blocks <= (img_size // block_size) ** 2.
        device (torch.device | str): Device for the returned tensor.

    Returns:
        torch.Tensor: Binary mask of shape (bs, g, g) with dtype torch.uint8,
        where g = img_size // block_size. A value of 1 marks a selected block,
        0 otherwise. To expand to pixel resolution (bs, img_size, img_size),
        upsample with:
            torch.kron(mask.float(), torch.ones(block_size, block_size, device=device)).byte()
    """
    g = img_size//block_size 
    total_blocks = g*g
    rand_indices = torch.argsort(torch.rand(bs, total_blocks, device=device), dim=1)[:,:n_blocks]
    masks = torch.zeros(bs, g, g, dtype=torch.uint8, device=device)
    rows = rand_indices//g 
    cols = rand_indices%g 
    batch_indices = torch.arange(bs, device=device).unsqueeze(1).expand(-1,n_blocks)
    masks[batch_indices, rows, cols] = 1 
    return masks

@torch.no_grad()
def point_block_mask(bs, ldmks, img_size, block_size,n_blocks):
    """
    Create a block mask guided by landmark proximity.

    Args:
        bs (int): Batch size.
        ldmks (torch.Tensor): Landmark coordinates of shape (bs, P, 2).
            Coordinates are expected in block-grid units [0, g-1], with
            g = img_size // block_size. If landmarks are in pixel units,
            convert with: torch.floor(ldmks / block_size).
        img_size (int): Image height/width in pixels. Must be divisible by `block_size`.
        block_size (int): Side length of a square block in pixels.
        n_blocks (int): Total number of blocks to select per image (approximately
            distributed across landmarks).

    Returns:
        torch.Tensor: Binary mask of shape (bs, img_size, img_size) with dtype
        torch.uint8. Ones indicate selected blocks expanded to pixel resolution.
    """
    device=ldmks.device
    g = img_size//block_size 

    P = ldmks.shape[1]

    coords = torch.arange(g, device=device)
    rr,cc = torch.meshgrid(coords, coords, indexing='ij')
    rr = rr.reshape(1,1,g,g).to(device)
    cc = cc.reshape(1,1,g,g).to(device)

    pr = ldmks[:,:,0].reshape(bs,P,1,1)
    pc = ldmks[:,:,1].reshape(bs,P,1,1)

    dist = torch.abs(rr-pr) + torch.abs(cc-pc)
    probs = gaussian_prob(dist, sigma=1.5).sum(dim=1)

    flat = probs.view(bs,P,-1)
    _, idx = torch.topk(flat,k=int(n_blocks//P), dim=2)

    mask = torch.zeros(bs,g*g,dtype=torch.uint8, device=device)
    base = torch.arange(bs, device=device).view(bs,1,1)
    base = base.expand(-1,P,int(n_blocks//P))
    mask[base.reshape(-1),idx.reshape(-1)] = 1 
    mask = mask.view(bs,g,g)

    return torch.kron(mask.float(),torch.ones(block_size,block_size, device=device)).byte()