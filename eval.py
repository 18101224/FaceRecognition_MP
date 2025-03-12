from models import kprpe_fer, load_ckpt_kprep
from dataset import FER
from torch.utils.data import DataLoader
import torch
from aligners import get_aligner
from utils import get_acc
from tqdm import tqdm
from utils import plot_confusion
from argparse import ArgumentParser, Namespace

def get_args():
    args = ArgumentParser()
    args.add_argument('--path')
    args.add_argument('--name')
    return args.parse_args()


config_path = 'rl/checkpoint/adaface_vit_base_kprpe_webface4m'

args = get_args()
path = args.path

data_arg = {'dataset_path':'data/AffectNet7', 'world_size':1}
data_arg = Namespace(**data_arg)
dataset = FER(data_arg,'checkpoint/quality',train=True)
loader = DataLoader(dataset,shuffle=False,batch_size=12)


device = torch.device('cuda')
model = kprpe_fer(config_path)
model.load_from_state_dict(args.path)
model = model.to(device)

aligner = get_aligner(cfg_path=config_path).to(device)


n = len(dataset)
result = 0
y_hats = []
ys = []

model.train()
aligner.eval()
embeds = torch.empty((1,512)).to(device)


with torch.no_grad():
    for img,label,q,c_q in tqdm(loader):
        img = img.to(device)
        _,ldmk,_,_,_,_ = aligner(img)
        label = label.to(device)
        embed,pred = model(img, ldmk)
        y_hat = torch.argmax(pred,dim=-1).cpu().reshape(-1).tolist()
        y_hats+=y_hat
        ys+=label.cpu().reshape(-1).tolist()
        bs = label.shape[0]
        acc = get_acc(pred,label)
        result += (bs/n)*acc
        norms = embed.norm(p=2,dim=-1,keepdim=True)
        embeds = torch.concat((embeds,embed/norms),dim=0)

    plot_confusion(y_hats,ys,args.name)

    print(result)
