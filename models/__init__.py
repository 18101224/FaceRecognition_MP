from .modules import Backbone as ir50_backbone
from .modules import get_QCS_model_single
from .modules import get_ssl
from .modules import vit_small_patch16
import sys
sys.path.extend('..')
from utils import *
from functools import partial
from .kp_rpe import get_kprpe_pretrained
from .utils import compute_class_spherical_means, slerp, calc_class_mean

__all__ = ['get_ir', 'kprpe_fer',
 'make_g_nets', 'IR50FPModel',
  'get_noise_model', 'ir50_backbone',
   'get_model', 'get_kprpe_pretrained',
   'get_mfnet', 'vit_small_patch16',
   'vit_base_patch16', 'vit_large_patch16',
    'vit_huge_patch14', 'compute_class_spherical_means', 'slerp', 'calc_class_mean', 'dim_dict']


model_dict = {
    'MoCov3': partial(get_ssl, model_type='MoCov3'),
    'Dinov2': partial(get_ssl, model_type='Dinov2'),
    'ir50': partial(ir50_backbone, checkpoint_path='checkpoint/ir50.pth'),
    'kprpe12m': partial(get_kprpe_pretrained, cfg_path='checkpoint/adaface_vit_base_kprpe_webface12m'),
    'kprpe4m': partial(get_kprpe_pretrained, cfg_path='checkpoint/adaface_vit_base_kprpe_webface4m'),
    'fmae_small': partial(vit_small_patch16, ckpt_path='checkpoint/FMAE_ViT_small.pth'),
    'Pyramid_ir50': partial(get_QCS_model_single, backbone_type='ir50', dim=768, num_classes=7),
}
dim_dict = {
    'ir50': (256, 512, 128),
    'MoCov3': (384, 1024, 512),
    'Dinov2': (384, 1024, 512),
    'kprpe12m': (512, 1024, 256),
    'kprpe4m': (512, 1024, 256),
    'fmae_small': (384, 1024, 512),
    'Pyramid_ir50': (768, 1024, 256)
}

def __getattr__(name):
    if name in {"IR50FPModel", "ImbalancedModel"}:
        from quantization.calibration.fp_model import IR50FPModel
        return IR50FPModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    
