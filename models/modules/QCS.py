from doctest import DocFileSuite
import torch 
from torch import nn 
from torch.nn import functional 
from timm.models.layers import trunc_normal_, DropPath 
from .vit import VisionTransformer, PatchEmbed
from functools import partial 
from .ir50 import Backbone
import sys;sys.path.extend('..')
from ..kp_rpe import get_kprpe_pretrained


__all__ = ['get_QCS_model', 'get_QCS_model_single']

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def get_QCS_model(backbone_type,dim, num_classes):
    backbone_dict={
        'ir50':(partial(Backbone,checkpoint_path='checkpoint/ir50.pth'),[64,128,256]),
        'kp_rpe':(partial(get_kprpe_pretrained,cfg_path='checkpoint/adaface_vit_base_kprpe_webface12m'),[512,1024,256])
    }
    backbone = backbone_dict[backbone_type][0]()
    return Pyramid(embed_dim=dim, num_classes=num_classes, dims=backbone_dict[backbone_type][1], backbone=backbone)

def get_QCS_model_single(backbone_type,dim, num_classes):
    backbone_dict={
        'ir50':(partial(Backbone,checkpoint_path='checkpoint/ir50.pth'),[64,128,256]),
        'kp_rpe':(partial(get_kprpe_pretrained,cfg_path='checkpoint/adaface_vit_base_kprpe_webface12m'),[512,1024,256])
    }
    backbone = backbone_dict[backbone_type][0]()
    return Pyramid_single(embed_dim=dim, num_classes=num_classes, dims=backbone_dict[backbone_type][1], backbone=backbone)


def Attn_QCS_SD(anchor, positive, neg1, neg2, k, alpha):
    '''
    anchor : B,N,C 
    positive : 2,B,N,C 
    neg1 : 2,2,B,N,C
    neg2 : 2,2,B,N,C
    '''

    B,N,C = anchor.shape 
    # num_positives, B,N,C

    neg1 = torch.tensor(neg1, device=anchor.device)
    neg2 = torch.tensor(neg2, device=anchor.device)
    k = k+1 

    # for anchor 

    S_p = torch.cdist(positive,anchor,p=2)
    S_p = S_p -torch.min(S_p)
    S_p = torch.max(S_p)-S_p 
    S_a = functional.normalize(S_p,p=2,dim=2)
    S_a = torch.sum(S_a,dim=1)
    S_p1 = S_p.transpose(1,2)
    S_p1 = functional.normalize(S_p1,p=2,dim=2)
    S_p1 = torch.sum(S_p1,dim=1)

    D1_1,D1_2 = torch.cdist(neg1,anchor.unsqueeze(0).repeat(2,1,1,1),p=2) # B n n
    D1_1 = D1_1 - torch.min(D1_1) 
    D1_2 = D1_2 - torch.min(D1_2)
    D1_1 = functional.normalize(D1_1,p=2,dim=2)
    D1_1 = torch.sum(D1_1,dim=1)
    D1_2 = functional.normalize(D1_2,p=2,dim=2)
    D1_2 = torch.sum(D1_2,dim=1)
    D2_1, D2_2 = torch.cdist(neg2, anchor.unsqueeze(0).repeat(2,1,1,1),p=2)
    D2_1 = D2_1 - torch.min(D2_1)
    D2_2 = D2_2 - torch.min(D2_2)
    D2_1 = functional.normalize(D2_1,p=2,dim=2)
    D2_1 = torch.sum(D2_1,dim=1)
    D2_2 = functional.normalize(D2_2,p=2,dim=2)
    D2_2 = torch.sum(D2_2,dim=1)
    map1 = S_a + k * (((D1_1+D1_2)/2)*alpha +(1-alpha)*((D2_1+D2_2)/2))

    D1_1,D1_2 = torch.cdist(neg1,positive.unsqueeze(0).repeat(2,1,1,1),p=2) # B n n
    D1_1 = D1_1 - torch.min(D1_1) 
    D1_2 = D1_2 - torch.min(D1_2)
    D1_1 = functional.normalize(D1_1,p=2,dim=2)
    D1_1 = torch.sum(D1_1,dim=1)
    D1_2 = functional.normalize(D1_2,p=2,dim=2)
    D1_2 = torch.sum(D1_2,dim=1)
    D2_1, D2_2 = torch.cdist(neg2,positive.unsqueeze(0).repeat(2,1,1,1),p=2)
    D2_1 = D2_1 - torch.min(D2_1)
    D2_2 = D2_2 - torch.min(D2_2)
    D2_1 = functional.normalize(D2_1,p=2,dim=2)
    D2_1 = torch.sum(D2_1,dim=1)
    D2_2 = functional.normalize(D2_2,p=2,dim=2)
    D2_2 = torch.sum(D2_2,dim=1)

    map2 = S_p1 + k * (((D1_1+D1_2)/2)*alpha +(1-alpha)*((D2_1+D2_2)/2))

    return map1.reshape(B,N,1), map2.reshape(B,N,1) 

class CrossAttention(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.proj = nn.Linear(embed_dim, embed_dim * 2)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=int(embed_dim * 4), act_layer=nn.GELU, drop=0)

        self.theta = nn.Parameter(torch.tensor(0.), requires_grad=True)
        self.alpha = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, anchor, positive, neg1, neg2):
        '''
        anchor : B,N,C 
        positive : B,N,C 
        neg1 : 2,B,N,C
        neg2 : 2,B,N,C
        return : B N C // 2 B N C 
        '''
        B, N, C = anchor.shape
        anchor = self.norm1(anchor)
        positive = self.norm1(positive)
        qk_x, v_x = self.proj(anchor).reshape(B,N,2,C).permute(2,0,1,3)
        qk_p, v_p = self.proj(positive).reshape(B,N,2,C).permute(2,0,1,3)
        negative_sequences1 = []
        for negative in neg1 : 
            negative = self.norm1(negative)
            qk, v = self.proj(negative).reshape(B, N, 2, C).permute(2, 0, 1, 3)
            negative_sequences1.append((qk,v,negative))

        negative_sequences2 = []
        for negative in neg2 : 
            negative = self.norm1(negative)
            qk, v = self.proj(negative).reshape(B, N, 2, C).permute(2, 0, 1, 3)
            negative_sequences2.append((qk,v,negative))

        k = torch.tanh(self.theta)
        alpha = torch.sigmoid(self.alpha)
        a_a, a_p = Attn_QCS_SD(qk_x, qk_p, torch.stack([qk for qk,_,_ in negative_sequences1]), torch.stack([qk for qk,_,_ in negative_sequences2]), k, alpha=alpha)
        n1_a, n1_p = Attn_QCS_SD(negative_sequences1[0][0], negative_sequences1[1][0], torch.stack([qk_x, qk_p]),torch.stack([qk for qk,_,_ in negative_sequences2]), k, alpha=alpha)
        n2_a, n2_p = Attn_QCS_SD(negative_sequences2[0][0], negative_sequences2[1][0], torch.stack([qk_x, qk_p]),torch.stack([qk for qk,_,_ in negative_sequences1]), k, alpha=alpha)

        anchor = (a_a*v_x) + anchor
        positive = (a_p*v_p) + positive
        negative_sequences1 = [(weight*value)*vector for weight, (_,value, vector) in zip([n1_a, n1_p],negative_sequences1)]
        negative_sequences2 = [(weight*value)*vector for weight, (_,value, vector) in zip([n2_a, n2_p],negative_sequences2)]

        anchor = self.mlp(self.norm2(anchor.clone())) + anchor
        positive = self.mlp(self.norm2(positive.clone())) + positive
        negative_sequences1 = [self.mlp(self.norm2(vector.clone())) + vector for vector in negative_sequences1]
        negative_sequences2 = [self.mlp(self.norm2(vector.clone())) + vector for vector in negative_sequences2]

        return anchor, positive, torch.stack(negative_sequences1), torch.stack(negative_sequences2)


class Pyramid(nn.Module):
    def __init__(self, embed_dim, num_classes,backbone, dims=[], ):
        super().__init__()
        self.num_classes = num_classes 

        self.ViT_base = VisionTransformer(depth=2, drop_ratio=0, embed_dim=embed_dim)
        self.ViT_cross = VisionTransformer(depth=1, drop_ratio=0, embed_dim=embed_dim)

        self.ir_back = backbone 

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=dims[i], out_channels=dims[i], kernel_size=3, stride=2, padding=1) for i in range(3)
        ]) # 1, 2, 3 
        self.embeds = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dims[0], 768, kernel_size=3, stride=2, padding=1),
                                     nn.Conv2d(768, 768, kernel_size=3, stride=2, padding=1))
        ]+[nn.Conv2d(dims[1],768,kernel_size=3,stride=2, padding=1)]+[PatchEmbed(img_size=14, patch_size=14, in_c=256, embed_dim=768)])


        self.cross_attentions = nn.ModuleList([
            CrossAttention(embed_dim) for _ in range(3)
        ])



    def process_convs(self,x):
        features = self.ir_back(x)[-1]
        features = [self.convs[i](features[i]) for i in range(3)]
        features = [self.embeds[i](features[i]).flatten(2).transpose(1, 2) if i<=1 else self.embeds[i](features[i]) for i in range(3)]
        out = torch.cat(features, dim=1)
        cls, patches = self.ViT_base(out)
        return cls, patches 


    def forward(self,x_a,x_p=None,x_n=None,x_n2=None):
        '''
        x_a : B,C,H,W
        x_p : B,C,H,W
        x_n : 2,B,C,H,W
        x_n2 : 2,B,C,H,W
        '''
        B,C,H,W = x_a.shape
        x_a, x_a_0 = self.process_convs(x_a)
        if x_p is None : 
            return x_a 
        x_p, x_p_0 = self.process_convs(x_p)
        B,N,D = x_p_0.shape
        N = N//3
        x_n, x_n_0 = self.process_convs(x_n.reshape(-1,C,H,W))
        x_n_0=x_n_0.reshape(B,2,3*N,D).permute(1,0,2,3)
        x_n2, x_n2_0 = self.process_convs(x_n2.reshape(-1,C,H,W)) 
        x_n2_0=x_n2_0.reshape(B,2,3*N,D).permute(1,0,2,3)  # 2,B,3N,C

        x_a_01, x_a_02, x_a_03 = torch.split(x_a_0, [N, N, N], dim=1)
        x_p_01, x_p_02, x_p_03 = torch.split(x_p_0, [N, N, N], dim=1)
        x_n_01, x_n_02, x_n_03 = torch.split(x_n_0, [N, N, N], dim=2) # 2 B N C 
        x_n2_01, x_n2_02, x_n2_03 = torch.split(x_n2_0, [N, N, N], dim=2)

        attn_a1, attn_p1, attn_n_1, attn_n2_1 = self.cross_attentions[0](x_a_01, x_p_01, x_n_01, x_n2_01)
        attn_a2, attn_p2, attn_n_2, attn_n2_2 = self.cross_attentions[1](x_a_02, x_p_02, x_n_02, x_n2_02)
        attn_a3, attn_p3, attn_n_3, attn_n2_3 = self.cross_attentions[2](x_a_03, x_p_03, x_n_03, x_n2_03)

        attn_a = torch.cat([attn_a1, attn_a2, attn_a3], dim=1)
        attn_p = torch.cat([attn_p1, attn_p2, attn_p3], dim=1)
        attn_n = torch.cat([attn_n_1, attn_n_2, attn_n_3], dim=2) # 2 B 3N C 
        attn_n2 = torch.cat([attn_n2_1, attn_n2_2, attn_n2_3], dim=2)

        x_a_0 = x_a_0 + attn_a
        x_p_0 = x_p_0 + attn_p
        x_n_0 = (x_n_0 + attn_n).reshape(-1,3*N,D)
        x_n2_0 = (x_n2_0 + attn_n2).reshape(-1,3*N,D) # 2B, 3N, C
        #2B,7
        out_a, _ = self.ViT_cross(x_a_0)
        out_p, _ = self.ViT_cross(x_p_0)
        out_n = self.ViT_cross(x_n_0)[0]
        out_n2 = self.ViT_cross(x_n2_0)[0] #vit 넣으면 알아서 저거 됨.classification 
    
        return x_a,x_p,x_n,x_n2, out_a, out_p, out_n, out_n2

class Pyramid_single(Pyramid):
    def __init__(self, embed_dim, num_classes, backbone, dims=[]):
        super().__init__(embed_dim, num_classes, backbone, dims)
    
    def forward(self,x, features=True, keypoint=None):
        cls_token, patches = self.process_convs(x)
        if features : 
            return cls_token, patches 
        else:
            return cls_token