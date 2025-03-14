from models import kprpe_fer, load_ckpt_kprep
from dataset import FER
from torch.utils.data import DataLoader
import torch
from aligners import get_aligner
from utils import get_acc
from tqdm import tqdm
from utils import plot_confusion
from argparse import ArgumentParser, Namespace
from analysis import plot_weight_tsne,analyze_weight_matrix
from torch import nn
def get_args():
    args = ArgumentParser()
    args.add_argument('--name')
    return args.parse_args()



if __name__ == '__main__':
    args = get_args()
    # data_arg = {'dataset_path':'data/AffectNet7', 'world_size':1}
    # data_arg = Namespace(**data_arg)
    # dataset = FER(data_arg,'checkpoint/quality',train=False)
    # loader = DataLoader(dataset,shuffle=False,batch_size=12)
    ckpt_path = 'checkpoint/only_margin_low'

    config_path = 'checkpoint/adaface_vit_base_kprpe_webface4m'
    model = kprpe_fer(ckpt_path,cos=True)
    device = torch.device('cuda')


    # change here

    model.load_from_state_dict(ckpt_path)
    model = model.to(device)
    model.train()
    weight = model.classifier.kernel.transpose(-1,-2)
    print('weight shape: ',weight.shape)
    normed_weight = nn.functional.normalize(weight,dim=0)
    print(torch.sum(normed_weight**2,dim=0))
    print('normed_weight norm',normed_weight.norm(p=2,dim=0))
    weight = model.classifier.kernel.transpose(-1,-2) # n_classes, dims
    weight = nn.functional.normalize(weight,dim=0)

    plot_weight_tsne(weight,save_path=f'{args.name}.png')
