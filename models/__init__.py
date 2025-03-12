from .ir50 import *
from copy import deepcopy
from .kp_rpe import *
import sys
sys.path.extend('..')
from utils import *



def get_ir():
    back = Backbone(50,0.0,'ir')
    back = load_pretrained_weights(back,torch.load('checkpoint/ir50.pth'))
    class IR(nn.Module):
        def __init__(self,backbone):
            super().__init__()
            self.backbone = backbone
            self.classifier = nn.Sequential(
                nn.AvgPool2d(kernel_size=14),
                nn.Flatten(),
                nn.BatchNorm1d(256),
                nn.Linear(256,128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128,7)
            )
            self.shadow = nn.ModuleDict({'backbone':deepcopy(self.backbone),
                                         'classifier':deepcopy(self.classifier)}).requires_grad_(False)
        def load_shadow(self):

            self.update(gamma=0)

        @torch.no_grad()
        def update(self, gamma):
            for param, shadow_param in zip(self.backbone.parameters(), self.shadow['backbone'].parameters()):
                shadow_param.data = gamma * shadow_param.data + (1 - gamma) * param.data
            for param, shadow_param in zip(self.classifier.parameters(), self.shadow['classifier'].parameters()):
                shadow_param.data = gamma * shadow_param.data + (1 - gamma) * param.data

        def apply_ema(self):
            for param, shadow_param in zip(self.backbone.parameters(),self.shadow['backbone'].parameters()):
                param.data.copy_(shadow_param.data)
            for param, shadow_param in zip(self.classifier.parameters(), self.shadow['classifier'].parameters()):
                param.data.copy_(shadow_param.data)

        def forward(self,x):
            _,_,x = self.backbone(x)
            return x, self.classifier(x)



    return IR(back)

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
             nn.Linear(512,256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
            ,nn.Linear(256,256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256,1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.net(x)

def load_ckpt(path,mode):
    if mode == 'r':
        ir = 'best_reward_c.pth'
        s = 'best_reward_s.pth'
        ct = 'best_reward_ct.pth'
    else:
        ir = 'best_acc_c.pth'
        s = 'best_acc_s.pth'
        ct = 'best_acc_ct.pth'

    ir_path = os.path.join(path,ir)
    selector_path = os.path.join(path,s)
    mean_path = os.path.join(path,ct)
    ir_ckpt = torch.load(ir_path,map_location=torch.device('cpu'))
    selector_ckpt = torch.load(selector_path,map_location=torch.device('cpu'))
    mean_ckpt = torch.load(mean_path,map_location=torch.device('cpu'))
    ir = get_ir()
    ir.load_state_dict(ir_ckpt)
    selector = Policy()
    selector.load_state_dict(selector_ckpt)
    return ir, selector, mean_ckpt[0]


class kprpe_fer(nn.Module):
    def __init__(self,cfg_path,token_path=None,force_download=False,cos=False):
        super().__init__()
        self.kp_rpe = get_kprpe_pretrained(cfg_path,token_path,force_download)
        self.classifier = Classifier() if not cos else CosClassifier()
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
