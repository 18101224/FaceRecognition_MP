# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer

__all__ = ['VisionTransformer','vit_small_patch16','vit_base_patch16','vit_large_patch16','vit_huge_patch14']

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        #input image size = 224 
        #patch size = 16 so 14*14 patches 

    def apply_mask(self, x, mask):
        # mask shape = (bs , 14, 14) mask = True if 
        # x shape = (bs, 14*14, 768) 
        B, N, C = x.shape
        # flatten mask to (B, N) and convert to boolean (1 -> keep, 0 -> drop)
        mask_flat = mask.view(B, -1)
        if mask_flat.shape[1] != N:
            raise ValueError(f"Mask length {mask_flat.shape[1]} does not match number of patches {N}.")

        mask_bool = mask_flat > 0.5

        # Ensure each sample keeps the same number of patches so we can stack
        kept_counts = mask_bool.sum(dim=1)
        if not torch.all(kept_counts == kept_counts[0]):
            raise ValueError("All samples in the batch must keep the same number of patches.")

        kept = int(kept_counts[0].item())
        # Collect kept patches per sample and stack back to (B, kept, C)
        out = x.new_empty((B, kept, C))
        for b in range(B):
            out[b] = x[b][mask_bool[b]]
        return out

    def forward_features(self, x, mask=None):
        B,_,h,_ = x.shape
        if h == 112 :
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        x = self.patch_embed(x) # flatten patches 

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        #x = self.pos_drop(x) # skip dropout 
        if mask is not None : 
            x = torch.cat((cls_tokens, self.apply_mask(x[:,1:], mask)),dim=1)
        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def load_ckpt(ckpt):
    return torch.load(ckpt,map_location='cpu',weights_only=False)['model']

def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    if getattr(kwargs,'ckpt_path',None) is not None :
        model.load_state_dict(load_ckpt(kwargs['ckpt_path']))
    return model


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    if getattr(kwargs,'ckpt_path',None) is not None :
        model.load_state_dict(load_ckpt(kwargs['ckpt_path']))
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    if getattr(kwargs,'ckpt_path',None) is not None :
        model.load_state_dict(load_ckpt(kwargs['ckpt_path']))
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    if getattr(kwargs,'ckpt_path',None) is not None :
        model.load_state_dict(load_ckpt(kwargs['ckpt_path']))
    return model