from .class_adaloss import *
from .instance_adaloss import *
from .label_noise import get_label_noise, including_margin
from .angleLoss import get_angle_loss
from .OOS_LNAAL import *
from .Imbalanced import *
from .FER import HCM
from .CL import KCL, Moco, compute_etf_loss, EKCL, get_cl_loss

__all__ = ['Proportion_loss', 'cosine_constant_margin_loss', 'get_label_noise', 
'including_margin', 'get_angle_loss', 'get_confidence_db', 'get_instant_margin', 
'apply_margin', 'BalSCL', 'BCLLoss', 'ECELoss', 'HCM', 'KCL', 'Moco', 'compute_etf_loss', 'EKCL', 'get_cl_loss']

class Proportion_loss:
    def __init__(self, labels, alpha,device):
        count = Counter(labels.tolist())
        array = []
        for key, value in count.items():
            array.append((int(key),value))
        array.sort(key = lambda x : x[0])
        self.array = array
        self.proportion = torch.tensor(array).float()[:,1].to(device) # num_class
        self.proportion_weights = self.get_adapt(alpha).to(device)
    def get_adapt(self,alpha=0):
        mean = self.proportion.mean()
        std = self.proportion.std()
        x = (self.proportion - mean) / std
        x = x + alpha
        x = torch.nn.functional.softmax(x)
        mx = torch.max(x)
        return (1+mx)-x

    def get_proportion_weights(self,label):
        return self.proportion_weights[label]

    def get_gamma(self,label):
        return 1/torch.pow(self.proportion[label],0.25)



def margin_logit(cos,j,angle,m,gamma):
    '''
    :param cos: argmax 값들만 취할것
    :param j:
    :param m:
    :param gamma:
    :return:
    '''

    thetas = torch.arccos(cos)

    cos = torch.cos(thetas-m*angle*(1-j)) - m*j*gamma
    return cos


def cosine_constant_margin_loss(preds, labels, m):
    margin = torch.zeros_like(preds, device=preds.device)
    margin[torch.arange(preds.shape[0]), labels] += m
    preds = preds - margin 
    return torch.nn.functional.cross_entropy(preds, labels)