from .modules import Backbone as ir50_backbone
from .modules import get_QCS_model_single
from .modules import get_ssl 
from .modules import vit_small_patch16
from torch import nn
import sys
sys.path.extend('..')
from utils import *
from functools import partial
from .kp_rpe import get_kprpe_pretrained
import torch 
from .utils import compute_class_spherical_means, slerp, calc_class_mean

__all__ = ['get_ir', 'kprpe_fer',
 'make_g_nets', 'ImbalancedModel',
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


class ImbalancedModel(nn.Module):
    # The model_dict dictionary maps model type strings to their corresponding constructor functions.
    # You can use it to dynamically select and instantiate a model based on a string key.
    def __init__(self, num_classes, model_type: str, feature_branch=False, feature_module=False,  
    regular_simplex=False, cos=True
    , learnable_input_dist=False, input_layer = False, freeze_backbone=False, remain_backbone=False,
    decomposition=False, img_size=112, use_bn = False, gap=False, scn=False, **kwargs):
        global model_dict, dim_dict
        super().__init__()
        # Use model_dict to get the constructor and instantiate the model
        self.backbone = model_dict[model_type]()
        dim_in, mid_dim, feat_dim = dim_dict[model_type]

        if feature_branch : 
            self.head = nn.Sequential(nn.Linear(dim_in, mid_dim), nn.BatchNorm1d(mid_dim), nn.ReLU(inplace=True), nn.Linear(mid_dim, feat_dim))
            self.head_fc = nn.Sequential(nn.Linear(dim_in, mid_dim), nn.BatchNorm1d(mid_dim), nn.ReLU(inplace=True), nn.Linear(mid_dim, feat_dim))
            self.bn = nn.BatchNorm1d(feat_dim) if use_bn else None


        self.feature_branch = feature_branch
        self.feature_module = feature_module

        
        regular_simplex = num_classes - 1 if regular_simplex else dim_in
        self.cos = cos 
        self.weight = torch.randn((regular_simplex,num_classes)).uniform_(-1,1).renorm(2,1,1e-5).mul_(1e5)
        self.weight = nn.Parameter(self.weight, requires_grad=True) # weight is dim, num_classes


        if learnable_input_dist:
            init_input_dist = torch.cat([torch.zeros((1,3)), torch.ones((1,3))], dim=0)
            self.input_dist = nn.Parameter(init_input_dist, requires_grad=True)
        if input_layer :
            raise ValueError('input_layer is not supported')
        self.freeze = freeze_backbone

        self.img_size=img_size
        self.gap = gap

        self.scn_weight = nn.Parameter(torch.randn((dim_in,1),requires_grad=True)) if scn else None 



    def init_weight(self, weight):
        self.weight = nn.Parameter(weight, requires_grad=True)
    
    def load_from_state_dict(self, ckpt_path, clear_weight=True):
        state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'),weights_only=False)['model_state_dict']
        self.load_state_dict(state_dict)
        if clear_weight : 
            self.weight.data = torch.randn((self.weight.shape[0], self.weight.shape[1])).uniform_(-1,1).renorm(2,1,1e-5).mul_(1e5)
        
    def get_kernel(self):
        if self.cos : 
            return torch.nn.functional.normalize(self.weight, dim=0, p=2) # dim, num_classes
        else:
            return self.weight

    def process_feature_branch(self,z, weight):
        w = self.head_fc(weight.T)
        z = self.head(z)
        if self.bn is not None:
            z = self.bn(z)
            w = self.bn(w)
        return nn.functional.normalize(z, dim=-1), nn.functional.normalize(w, dim=-1)

    def forward(self, x, features=False, keypoint=None, wo_branch=False, featuremap=False):
        '''
        returns : backbone_feature, rotated_feature, logit
        '''
        if self.img_size == 224 : 
            x = nn.functional.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)

        if keypoint is not None : 
            keypoint = keypoint.clone().detach()
            
        if hasattr(self, 'input_dist') and self.input_dist is not False:
            mean = self.input_dist[0].reshape(1,3,1,1)
            std = torch.clamp(self.input_dist[1], min=1e-6).reshape(1,3,1,1)
            x = (x - mean) / (std+1e-8)

        to_with = torch.enable_grad if not self.freeze else torch.no_grad

        with to_with():
            if keypoint is not None:
                if featuremap or self.gap :
                    _, featuremaps = self.backbone(x, keypoint, featuremap=True)
                    z = featuremaps.mean(dim=1)
                    z = torch.nn.functional.normalize(z, dim=-1, eps=1e-6)
                    weight = self.get_kernel()
                    if featuremap :
                        return z@weight, featuremaps, weight.T
                    else:
                        return z@weight
                else:
                    z = self.backbone(x, keypoint)
            else:
                z = self.backbone(x)
        
        if self.scn_weight is not None : 
            weight = self.get_kernel()
            logit = z @ weight 
            if self.training : 
                return logit, torch.nn.functional.sigmoid(z @ self.scn_weight ), None 
            else:
                return logit 
                
        if isinstance(z, tuple):
            z = z[0]

        z = nn.functional.normalize(z, dim=-1, eps=1e-6) if self.cos else z

        z_ = nn.functional.normalize(self.feature_module(z), dim=-1, eps=1e-6) if self.feature_module is not False else z 


        ### if adversarial training 
        if getattr(self, 'decomposition', None) is not None:
            z1, z2 = self.decomposition(z_)
            z1 = nn.functional.normalize(z1, dim=-1)
            z2 = nn.functional.normalize(z2, dim=-1)
            logit = z1 @ self.get_kernel()
            if features :
                return logit, z1, z2
            else:  
                return logit

        
        weight = self.get_kernel() # dim, num_classes

        logit = z_ @ weight

        if featuremap : 
            return logit, featuremaps, weight.T
        # if CL training 
        if features : 
            processed_feat, centers = self.process_feature_branch(z_, weight) if self.feature_branch else (z_, weight.T)
            if not wo_branch :
                return logit, processed_feat , centers
            else: # when wo_branch is True 
                return logit, processed_feat, centers, z_
        else:
            return logit 
    