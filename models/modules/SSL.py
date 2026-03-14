import torch
import torch.nn as nn
import timm 
from transformers import AutoModel

__all__ = ['SSL', 'get_ssl']

class SSL(nn.Module):
    def __init__(self, model_type):
        super().__init__()
        if model_type == 'Dinov2':
            self.model = AutoModel.from_pretrained("facebook/dinov2-small")
        elif model_type == 'MoCov3':
            self.model = timm.create_model('vit_small_patch16_224',
            pretrained=False,
            num_classes=0,
            global_pool="")
            ckpt = torch.load('checkpoint/mocov3.pth.tar',weights_only=False)
            sd = ckpt.get('state_dict',ckpt)
            def strip_prefix(s, p):
                return s[len(p):] if s.startswith(p) else s
            new_sd = {}
            for k, v in sd.items():
                k = strip_prefix(k, "module.")
                k = strip_prefix(k, "base_encoder.")
                k = strip_prefix(k, "module.base_encoder.")
                # 분류 헤드 있으면 버림
                if k.startswith(("head.", "fc.", "classifier.")):
                    continue
                new_sd[k] = v
            self.model.load_state_dict(new_sd, strict=False)

    def forward(self,x,**kwargs):
        out = self.model(x)
        if hasattr(out, 'last_hidden_state'):
            return out.last_hidden_state[:, 0]
        return out[:, 0]


def get_ssl(model_type):
    return SSL(model_type)