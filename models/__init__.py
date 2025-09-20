from .modules import Backbone as ir50_backbone
from .modules import Parital_Backbone 
from .modules import DeepComplexModule, ResidualModule
from .modules import resnet32_backbone, resnet50_backbone, resnext50_backbone
from .modules import e2_resnet32, e2_resnext50
from .modules import OrthogonalDecomposer
from .modules import get_QCS_model_single
#from .modules import Combiner, multi_network, multi_network_MOCO
from copy import deepcopy
from torch import nn
import sys
sys.path.extend('..')
from utils import *
import os
import torchvision.models as tv_models
from torch.nn.parallel import DistributedDataParallel as DDP
from functools import partial
from .kp_rpe import get_kprpe_pretrained
import torch 
from .modules import get_mfnet, vit_small_patch16, vit_base_patch16, vit_large_patch16, vit_huge_patch14
from .utils import compute_class_spherical_means, slerp, calc_class_mean

__all__ = ['get_ir', 'kprpe_fer',
 'make_g_nets', 'ImbalancedModel',
  'get_noise_model', 'ir50_backbone',
   'get_model', 'get_kprpe_pretrained', 
   'get_mfnet', 'vit_small_patch16', 
   'vit_base_patch16', 'vit_large_patch16',
    'vit_huge_patch14', 'compute_class_spherical_means', 'slerp', 'calc_class_mean', 'dim_dict']


model_dict = {
    'resnet32_64d': partial(resnet32_backbone, factor=1),
    'resnet32_128d': partial(resnet32_backbone, factor=2),
    'resnet50': partial(resnet50_backbone, pretrained=False),
    'resnext50': partial(resnext50_backbone, pretrained=False),
    'ir50': partial(ir50_backbone, checkpoint_path='checkpoint/ir50.pth'),
    'e2_resnet32': e2_resnet32,
    'e2_resnext50': e2_resnext50,
    'kp_rpe': partial(get_kprpe_pretrained, cfg_path='checkpoint/adaface_vit_base_kprpe_webface12m'),
    **{
        f'ir50_{i}': partial(Parital_Backbone, checkpoint_path='checkpoint/ir50.pth', to_What=i) for i in range(1,4)
    },
    'fmae_small': partial(vit_small_patch16, ckpt_path='checkpoint/FMAE_ViT_small.pth'),
    'Pyramid_ir50': partial(get_QCS_model_single, backbone_type='ir50', dim=768, num_classes=7),
}
dim_dict = {
    'resnet32_64d': (64, 512, 128),
    'resnet32_128d': (128, 512, 128),
    'resnet50': (2048, 2048, 1024),
    'resnext50': (2048, 2048, 1024),
    'ir50': (256, 512, 128),
    'e2_resnet32': (256, 256, 128),
    'e2_resnext50': (2048, 2048, 1024),
    'kp_rpe': (512, 1024, 256), 
    **{
        f'ir50_{i}': (dim_in, mid_dim, feat_dim) for i, dim_in, mid_dim, feat_dim in zip(list(range(1,4)), [64,128,256], [128,512,256], [256,1048,512])
    },
    'fmae_small': (384, 1024, 512),
    'Pyramid_ir50': (768, 1024, 256)
}

if True : 
    from .kp_rpe import *
    class kprpe_fer(nn.Module):
        def __init__(self,cfg_path,token_path=None,force_download=False,cos=False):
            super().__init__()
            self.kp_rpe = get_kprpe_pretrained(cfg_path,token_path,force_download)
            
            self.cos = cos

        def forward(self,x,keypoint):
            embed = self.kp_rpe(x,keypoint)
            if not self.cos :
                out = self.classifier(embed)
                return embed, out
            else:
                margin, out = self.classifier(embed)
                return embed,out,margin

        def load_from_state_dict(self,path):
            cls_pt = torchload(os.path.join(path,'classifier.pt'),weights_only=True)
            self.classifier.load_state_dict(cls_pt)
            kp_rpe_pt = os.path.join(path,'model.pt')
            self.kp_rpe.load_state_dict_from_path(kp_rpe_pt)
            self.kp_rpe.input_color_flip = False


        def train_PatchEmbed(self):
            self.kp_rpe.train()
            self.classifier.train()
            for p in self.kp_rpe.parameters():
                p.requires_grad_(False)
            for p in self.classifier.parameters():
                p.requires_grad_(False)


    def get_model(model_config):
        from .vit_kprpe import load_model as load_vit_kprpe_model
        model = load_vit_kprpe_model(model_config)
        if model_config.start_from:
            model.load_state_dict_from_path(model_config.start_from)

        if model_config.freeze:
            for param in model.parameters():
                param.requires_grad_(False)
        return model


def get_residual_module(in_features, num_hidden_layers):
    if num_hidden_layers == 0:
        return None, None, None
    else:
        blocks = []
        norms = []
        activations = []
        for i in range(num_hidden_layers):
            blocks.append(
                nn.Sequential(
                    nn.Linear(in_features, in_features),
                    nn.BatchNorm1d(in_features),
                    nn.LeakyReLU(),
                    nn.Dropout(0.3)
                )
            )
            norms.append(nn.BatchNorm1d(in_features))
            activations.append(nn.SiLU())
        return nn.ModuleList(blocks), nn.ModuleList(norms), nn.ModuleList(activations)


def get_imgnet_resnet():
    """
    Loads an ImageNet-pretrained ResNet model with more than 150 layers from torchvision.
    Returns:
        model (nn.Module): Pretrained ResNet-152 model.
    """
    model = tv_models.resnet101(weights="IMAGENET1K_V1")
    return model


class CosClassifier(nn.Module):
    def __init__(self,num_classes, backbone, dim, num_hidden_layers):
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        self.blocks, self.norms, self.activations = get_residual_module(dim, num_hidden_layers)
        weight = torch.randn((num_classes, dim))
        weight = torch.nn.functional.normalize(weight, p=2, dim=1)
        self.weight = nn.Parameter(
            weight.transpose(-1, -2), requires_grad=True
        ) # dim, num_class

    def pass_blocks(self,x):
        for block, norm, act in zip(self.blocks, self.norms, self.activations):
            res = x.clone()
            x = block(x)
            x = x + res
            x = norm(x)
            x = act(x)
        return x 
    
    def get_kernel(self):
        return torch.nn.functional.normalize(self.weight, dim=0, p=2) # dim, num_classes
    
    def forward(self,x, keypoint=None, features=False):
        if keypoint is not None:
            x = self.backbone(x,keypoint)
        else:
            x = self.backbone(x)
        if self.blocks is not None:
            x = self.pass_blocks(x)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        kernel = self.get_kernel() # dim, num_class
        cos = x @ kernel  # bs, num_classes
        if features : 
            return x, cos 
        else: 
            return cos
        
def get_feature_module(feature_module, dim_in, depth, regular_simplex=False, num_classes=None):
    if feature_module == 'deepcomplex':
        return DeepComplexModule(dim_in, depth, regular_simplex, num_classes)
    elif feature_module == 'residual':
        return ResidualModule(dim_in, depth)
    else:
        raise ValueError(f"Invalid feature module: {feature_module}")

class Perpendicular(nn.Module):
    def __init__(self, model_type):
        super().__init__()


class ImbalancedModel(nn.Module):
    # The model_dict dictionary maps model type strings to their corresponding constructor functions.
    # You can use it to dynamically select and instantiate a model based on a string key.

    def __init__(self, num_classes, model_type: str, feature_branch=False, feature_module=False,  
    regular_simplex=False, cos=True
    , learnable_input_dist=False, input_layer = False, freeze_backbone=False, remain_backbone=False,
    decomposition=False, img_size=112):
        global model_dict, dim_dict
        if model_type not in model_dict:
            raise ValueError(f"Invalid model type: {model_type}")
        super().__init__()
        # Use model_dict to get the constructor and instantiate the model
        if model_type in ['ir50_1', 'ir50_2', 'ir50_3']:
            self.backbone = model_dict[model_type](remain_backbone=remain_backbone)
            dim_in, mid_dim, feat_dim = dim_dict['ir50'] if remain_backbone else dim_dict[model_type]
            print(f'remain : {remain_backbone} {model_type} dim : {dim_in}, {mid_dim}, {feat_dim}')
        else:
            self.backbone = model_dict[model_type]()
            dim_in, mid_dim, feat_dim = dim_dict[model_type]

        if feature_branch : 
            self.head = nn.Sequential(nn.Linear(dim_in, mid_dim), nn.BatchNorm1d(mid_dim), nn.ReLU(inplace=True), nn.Linear(mid_dim, feat_dim))
            self.head_fc = nn.Sequential(nn.Linear(dim_in, mid_dim), nn.BatchNorm1d(mid_dim), nn.ReLU(inplace=True), nn.Linear(mid_dim, feat_dim))


        self.feature_branch = feature_branch
        self.feature_module = feature_module

        if self.feature_module : 
            feature_module, depth = feature_module.split('_')
            self.feature_module = get_feature_module(feature_module, dim_in, int(depth))
        
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

        self.decomposition = OrthogonalDecomposer(dim_in) if decomposition=='Cayley' else None

        self.img_size=img_size
    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
    

    def get_kernel(self):
        if self.cos : 
            return torch.nn.functional.normalize(self.weight, dim=0, p=2) # dim, num_classes
        else:
            return self.weight
        
    def forward(self, x, features=False, keypoint=None ):
        '''
        returns : backbone_feature, rotated_feature, logit
        '''
        if self.img_size == 224 : 
            x = nn.functional.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)

        if hasattr(self, 'input_dist') and self.input_dist is not False:
            mean = self.input_dist[0].reshape(1,3,1,1)
            std = torch.clamp(self.input_dist[1], min=1e-6).reshape(1,3,1,1)
            x = (x - mean) / (std+1e-8)

        to_with = torch.enable_grad if not self.freeze else torch.no_grad

        with to_with():
            if keypoint is not None:
                z = self.backbone(x, keypoint) 
            else:
                z = self.backbone(x)
        
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

        # if CL training 
        if features : 
            centers = nn.functional.normalize(self.head_fc(weight.T), dim=-1) if self.feature_branch else weight.T
            processed_feat = nn.functional.normalize(self.head(z_), dim=-1) if self.feature_branch else z_
            return logit, processed_feat , centers 
        else:
            return logit 
    
    def analysis(self, x):
        '''
        returns : backbone_feat, cls_feat, bcl_feat, center_feat
        '''
        backbone_feat = self.backbone(x)
        
        cls_feat = torch.nn.functional.normalize(self.feature_module(backbone_feat), dim=1) if self.feature_module is not False else backbone_feat

        bcl_feat = torch.nn.functional.normalize(self.head(backbone_feat), dim=1) if self.feature_branch else backbone_feat
        
        center_feat = torch.nn.functional.normalize(self.head_fc(self.weight.T), dim=1) if self.feature_branch else self.weight.T

        logit = cls_feat @ self.get_kernel()

        return backbone_feat, cls_feat, bcl_feat, center_feat, logit




def get_sims(kernel):
    # kernel shape : dim, n_classes
    sims = kernel.T @ kernel 
    i, j = torch.triu_indices(kernel.shape[1], kernel.shape[1], offset=1)
    up = sims[i,j].reshape(-1)
    return up

def get_noise_model(args, pretrained=False):
    if args.dataset_name in ['AffectNet', 'RAF-DB']:  # fer
        model = CosClassifier(
            num_classes=7,
            dim=512,
            num_hidden_layers=4,
            backbone=deepcopy(get_kprpe_pretrained(args.kprpe_ckpt_path, token_path=False, force_download=False))
        )
    else:
        from .modules.resnet_backbone import get_resnet
        backbone = get_resnet(args.architecture, pretrained=pretrained)
        model = CosClassifier(
            num_classes=args.num_classes,
            dim=2048,
            backbone=backbone,
            num_hidden_layers=4
        )
    return model

def make_g_nets(args, device, freeze=False, pretrained=False):
    model = get_noise_model(args, pretrained=pretrained)
    g_nets = []
    for _ in range(args.n_folds):
        m = deepcopy(model)
        m = m.to(device)
        if args.world_size > 1 and not freeze:
            m = DDP(module=m, device_ids=[device], find_unused_parameters=True)
        g_nets.append(m)
    return g_nets

def load_g_nets(g_nets, ckpt_path, device):
    from glob import glob 
    ckpts = sorted(glob(os.path.join(ckpt_path, 'g_net_*.pt')))
    for g_net, ckpt in zip(g_nets, ckpts):
        g_net.load_state_dict(torch.load(ckpt, map_location=torch.device(device)))
    return g_nets


def get_model(args):
    '''
    args should have dataset_name, model_type, cos, feature_module, feature_branch
    '''
    if 'cifar' in args.dataset_name:
        n_c = 100 if '100' in args.dataset_name else 10
        model = ImbalancedModel(cos=args.cos, num_classes=n_c, model_type=args.model_type, feature_module=args.feature_module, feature_branch=args.feature_branch)
        return model
    elif 'imagenet_lt' == args.dataset_name:
        model = ImbalancedModel(cos=args.cos, num_classes=1000, model_type=args.model_type)
        return model
    elif 'inat' == args.dataset_name:
        model = ImbalancedModel(cos=args.cos, num_classes=8142, model_type=args.model_type)
        return model
    elif 'RAF-DB' == args.dataset_name or 'AffectNet' == args.dataset_name:
        model = ImbalancedModel(cos=args.cos, num_classes=7, model_type=args.model_type, feature_module=args.feature_module, feature_branch=args.feature_branch)
        return model
    else:
        raise ValueError(f'Dataset {args.dataset_name} not supported')
