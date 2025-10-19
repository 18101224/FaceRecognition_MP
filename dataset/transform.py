from torchvision import transforms 
import torch
import torch.nn.functional as F
import random
import numpy as np
from .augmentations import ImageNetPolicy, CIFAR10Policy, Cutout, GaussianBlur
from PIL import ImageFilter
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from .augmentations import rand_augment_transform
from copy import deepcopy
import albumentations as A


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

def get_fer_transforms(train,model_type):
    mean=[0.485, 0.456, 0.406] if 'kprpe' not in model_type else [0.5] * 3
    std=[0.229, 0.224, 0.225] if 'kprpe' not in model_type else [0.5] * 3
    if train :
        # return transforms.Compose([
        #     transforms.Resize((112, 112)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        #     transforms.RandomRotation(15),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=mean, std=std)
        # ])
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
            transforms.Normalize(mean=mean, std=std)
        ]
        # 필요 시 CutOut/Erase/Crop 증강 더 추가 가능 (landmark ROI 안전 검사 필요)
        # 예) custom cutout(눈·입 방어)는 albumentations나 직접 커스텀 함수로 구현
        return transforms.Compose(pipeline)
    else : 
        return transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

def get_multi_view_transforms(args, train, model_type):
    mean=[0.485, 0.456, 0.406] if 'kprpe' not in model_type else [0.5] * 3
    std=[0.229, 0.224, 0.225] if 'kprpe' not in model_type else [0.5] * 3
    mean = torch.tensor(mean)
    std = torch.tensor(std)
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
    if args.use_view and train : 
        return [get_fer_transforms(train, model_type), deepcopy(view_tf), deepcopy(view_tf)]
    else:
        return get_fer_transforms(train, model_type)


def gaussian_prob(distance, sigma=1.5):
    return torch.exp(-0.5 * (distance / sigma) ** 2)


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
    return torch.kron(masks.float(), torch.ones(block_size, block_size, device=device)).byte() 

@torch.no_grad()
def point_block_mask(bs, ldmks, img_size, block_size, n_blocks, sigma=1.5, eye_sigma_scale=2.0):
    """
    Args:
        ldmks: (bs, P, 2) in normalized coordinates (0..1). Each point is the
               keypoint location in normalized image space and will be mapped
               to the coarse block grid of size g = img_size // block_size.
        img_size: int, image height/width in pixels (assumed square).
        block_size: int, side length of block in pixels.
        n_blocks: int, number of blocks to select per image (0..g*g).
        sigma: float, base Gaussian std in grid units.
        eye_sigma_scale: float, multiplicative factor applied to sigma for
            eye keypoints (indices 0 and 1) to broaden their influence.
    Returns:
        (bs, img_size, img_size) uint8 binary mask, with exactly `n_blocks`
        blocks set to 1 per image (unless n_blocks > g*g, then clamped).
    """
    device = ldmks.device
    g = img_size // block_size
    if g <= 0:
        return torch.zeros(bs, img_size, img_size, dtype=torch.uint8, device=device)

    # Clamp n_blocks to grid capacity
    max_blocks = g * g
    n_blocks = int(max(0, min(int(n_blocks), max_blocks)))
    # Handle degenerate cases early
    if n_blocks == 0 or ldmks.numel() == 0:
        return torch.zeros(bs, img_size, img_size, dtype=torch.uint8, device=device)

    P = ldmks.shape[1]
    if P == 0:
        return torch.zeros(bs, img_size, img_size, dtype=torch.uint8, device=device)
    # Grid coordinates (flattened)
    coords = torch.arange(g, device=device, dtype=ldmks.dtype)
    rr, cc = torch.meshgrid(coords, coords, indexing='ij')            # (g,g)
    rr = rr.reshape(1, 1, g * g)
    cc = cc.reshape(1, 1, g * g)

    # Landmarks: support normalized (0..1) or pixel (0..img_size-1)
    # Heuristic: if any value > 1.5, treat as pixel coordinates
    is_pixel = (ldmks.max() > 1.5)
    if is_pixel:
        # Note: rr compares with row (y), cc with col (x)
        pr = (ldmks[:, :, 1].clamp(0, img_size - 1) / block_size).unsqueeze(-1)  # (bs,P,1)
        pc = (ldmks[:, :, 0].clamp(0, img_size - 1) / block_size).unsqueeze(-1)  # (bs,P,1)
    else:
        pr = (ldmks[:, :, 1].clamp(0, 1) * (g - 1)).unsqueeze(-1)     # (bs,P,1)
        pc = (ldmks[:, :, 0].clamp(0, 1) * (g - 1)).unsqueeze(-1)     # (bs,P,1)


    # Per-point Gaussian probability over grid cells (with larger sigma for eyes)
    dist2 = (rr - pr) ** 2 + (cc - pc) ** 2                           # (bs,P,g*g)
    sigma_pp = torch.full((bs, P, 1), float(sigma), device=device, dtype=ldmks.dtype)
    if P >= 2:
        sigma_pp[:, 0, 0] = float(sigma) * float(eye_sigma_scale)
        sigma_pp[:, 1, 0] = float(sigma) * float(eye_sigma_scale)
    probs = torch.exp(-0.5 * (dist2 / (sigma_pp ** 2)))               # (bs,P,g*g)

    # Vectorized balanced allocation: enforce equal quota per keypoint
    K = g * g
    mask_flat = torch.zeros(bs, K, dtype=torch.uint8, device=device)  # (bs,K)
    base = n_blocks // P
    rem = n_blocks % P
    # per-point remaining quota (distribute the remainder to the first `rem` keypoints)
    remaining_pp = torch.full((bs, P), base, device=device, dtype=torch.long)
    if rem > 0:
        extra = torch.zeros(P, dtype=torch.long, device=device)
        extra[:rem] = 1
        remaining_pp = remaining_pp + extra.view(1, P)

    # Iterate at most max quota steps; break early if all done
    max_steps = int(base + (1 if rem > 0 else 0))
    for _ in range(max_steps):
        if torch.all(remaining_pp <= 0):
            break
        # Mask out already selected blocks
        cur_probs = probs.masked_fill(mask_flat.bool().unsqueeze(1), float('-inf'))  # (bs,P,K)
        # Mask out keypoints that have satisfied their quota
        cur_probs = cur_probs.masked_fill((remaining_pp <= 0).unsqueeze(-1), float('-inf'))
        # Best candidate per keypoint
        chosen = torch.argmax(cur_probs, dim=2)                       # (bs,P)
        # First occurrence across keypoints to avoid within-step duplicates
        one_hot = F.one_hot(chosen, num_classes=K).to(cur_probs.dtype)  # (bs,P,K)
        csum = torch.cumsum(one_hot, dim=1)
        first_mask_p = (one_hot > 0) & (csum == 1)                    # (bs,P,K)
        # We only take from keypoints that still have remaining quota this step
        take_mask = (remaining_pp > 0) & first_mask_p.any(dim=2)       # (bs,P)
        step_keep = (first_mask_p & take_mask.unsqueeze(-1)).any(dim=1)  # (bs,K)
        # Update selection and remaining counts
        newly_selected = step_keep & ~mask_flat.bool()
        num_new = newly_selected.sum(dim=1)
        mask_flat = (mask_flat.bool() | step_keep).to(torch.uint8)
        # Decrement per-point remaining where we picked
        # Determine which keypoints successfully placed a block
        success_pp = (first_mask_p & take_mask.unsqueeze(-1)).any(dim=2).to(torch.long)  # (bs,P)
        remaining_pp = remaining_pp - success_pp

    mask_grid = mask_flat.view(bs, g, g)                              # (bs,g,g)
    # Expand to pixel resolution
    return torch.kron(mask_grid.float(), torch.ones(block_size, block_size, device=device)).byte()

@torch.no_grad()
def apply_mask(img, mask, block_size, mask_vector=None):
    '''
    img : (bs, 3, h, w)
    mask : (bs, h, w) pixel-level 0/1 mask aligned to blocks
    block_size : int block side in pixels
    mask_vector : (3, block_size, block_size) or None. If provided, replace
                  selected blocks with this vector. Else, multiply image by mask.
    '''
    bs, _, h, w = img.shape
    device = img.device
    if mask_vector is None:
        return img * mask.unsqueeze(1).float()
    g = h // block_size
    # Coarse block mask (bs,g,g)
    coarse = mask.view(bs, g, block_size, g, block_size).amax(dim=(2,4))
    # Tile learnable block over selected cells
    block_vec = mask_vector.view(1, 3, 1, 1, block_size, block_size).to(img.dtype).to(device)
    tiled = torch.where(
        coarse.view(bs, 1, g, g, 1, 1).bool(),
        block_vec,
        torch.zeros(1, 3, 1, 1, block_size, block_size, device=device, dtype=img.dtype)
    )
    tiled = tiled.permute(0,1,2,4,3,5).contiguous().view(bs, 3, g*block_size, g*block_size)
    # Pixel-level mask expanded to channels
    m = mask.unsqueeze(1).float()
    return img * (1.0 - m) + tiled * m
    
@torch.no_grad()
def masking_pair(img, ldmk, n_blocks, block_size=7, mask_vector=None, sigma=1.5, eye_sigma_scale=3.0, anchor_mask=False):
    """
    Produce anchor and negative masked images.

    Args:
        img: (bs, 3, H, W) float tensor.
        ldmk: (bs, P, 2) keypoints (normalized (0..1) or pixel coords).
        n_blocks: int, number of blocks per image.
        block_size: int, pixel size for each block.

    Returns:
        anchor_img, neg_img
    """
    bs, _, h, _ = img.shape
    device = img.device
    u_mask = random_masking(bs=bs, img_size=h, block_size=block_size, n_blocks=n_blocks, device=device)
    l_mask = point_block_mask(bs=bs, ldmks=ldmk, img_size=h, n_blocks=n_blocks, block_size=block_size, sigma=sigma, eye_sigma_scale=eye_sigma_scale)
    if mask_vector is None:
        # Multiplicative masking (legacy)
        anchor_img = img * u_mask.unsqueeze(1) if  anchor_mask else img 
        neg_img = img * (1-l_mask).unsqueeze(1)
    else:
        # Replace selected blocks with the learnable vector.
        anchor_img = apply_mask(img, u_mask, block_size=block_size, mask_vector=mask_vector) if  anchor_mask else img 
        # For negatives, replace keypoint blocks directly (do not invert)
        neg_img = apply_mask(img, l_mask, block_size=block_size, mask_vector=mask_vector)
    return anchor_img, neg_img