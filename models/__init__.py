from .ir50 import Backbone as ir50_backbone
from copy import deepcopy
from .kp_rpe import *
import sys
sys.path.extend('..')
from utils import *
from .resmodule import Residual
import torchvision.models as tv_models
from .resnet_backbone import resnet32_backbone, resnet50_backbone, resnext50_backbone
from torch.nn.parallel import DistributedDataParallel as DDP
from functools import partial
from .feature_module import DeepComplexModule, ResidualModule

__all__ = ['get_ir', 'kprpe_fer', 'make_g_nets', 'ImbalancedModel', 'get_noise_model', 'ir50_backbone']


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
        return ResidualModule(dim_in, depth, regular_simplex, num_classes)
    else:
        raise ValueError(f"Invalid feature module: {feature_module}")

class ImbalancedModel(nn.Module):
    # The model_dict dictionary maps model type strings to their corresponding constructor functions.
    # You can use it to dynamically select and instantiate a model based on a string key.
    model_dict = {
        'resnet32': resnet32_backbone,
        'resnet50': partial(resnet50_backbone, pretrained=False),
        'resnext50': partial(resnext50_backbone, pretrained=False),
        'ir50': partial(ir50_backbone, checkpoint_path='../checkpoint/ir50.pth')
    }
    dim_dict = {
        'resnet32': (64, 512, 128),
        'resnet50': (2048, 2048, 1024),
        'resnext50': (2048, 2048, 1024),
        'ir50': (256, 512, 128)
    }
    def __init__(self, num_classes, model_type: str, feature_branch=True, feature_module=False, regular_simplex=False):
        if model_type not in self.model_dict:
            raise ValueError(f"Invalid model type: {model_type}")
        super().__init__()
        # Use model_dict to get the constructor and instantiate the model
        self.backbone = self.model_dict[model_type]()
        dim_in, mid_dim, feat_dim = self.dim_dict[model_type]

        if feature_branch : 
            self.head = nn.Sequential(nn.Linear(dim_in, mid_dim), nn.BatchNorm1d(mid_dim), nn.ReLU(inplace=True), nn.Linear(mid_dim, feat_dim))
            self.head_fc = nn.Sequential(nn.Linear(dim_in, mid_dim), nn.BatchNorm1d(mid_dim), nn.ReLU(inplace=True), nn.Linear(mid_dim, feat_dim))



        self.feature_branch = feature_branch
        self.feature_module = feature_module

        if self.feature_module : 
            feature_module, depth = feature_module.split('_')
            self.feature_module = get_feature_module(feature_module, dim_in, int(depth), regular_simplex, num_classes)
        
        regular_simplex = num_classes - 1 if regular_simplex else dim_in
        self.weight = torch.randn((regular_simplex,num_classes)).uniform_(-1,1).renorm(2,1,1e-5).mul_(1e5)
        self.weight = nn.Parameter(self.weight, requires_grad=True) # weight is dim, num_classes

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)


    def get_kernel(self):
        return torch.nn.functional.normalize(self.weight, dim=0, p=2) # dim, num_classes
        
    def forward(self, x, features=False):
        '''
        returns : feat_mlp, logits, projected_centers
        '''
        feat = self.backbone(x)
        feat = torch.nn.functional.normalize(feat, dim=1)

        logit_feat = torch.nn.functional.normalize(self.feature_module(feat), dim=1) if self.feature_module is not False else feat 
            

        kernel = self.get_kernel()
        logits = logit_feat @ kernel
        weight = self.weight.T

         # input is num_classes, dim 
        if features:
            if self.feature_branch :
                centers_logits = nn.functional.normalize(self.head_fc(weight), dim=1) # num_classes, dim 
                processed_feat =  nn.functional.normalize(self.head(logit_feat), dim=1) 
                feat = processed_feat
            else:
                centers_logits = weight
                processed_feat = logit_feat
            return feat, processed_feat, logits, centers_logits
        else:
            return logits
    
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
        from .resnet_backbone import get_resnet
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