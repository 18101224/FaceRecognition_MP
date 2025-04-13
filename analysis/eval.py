import sys
sys.path.extend('..')
from models import kprpe_fer, load_ckpt_kprep
from dataset import FER
from torch.utils.data import DataLoader
import torch
from aligners import get_aligner
from utils import get_acc
from tqdm import tqdm

from utils import plot_confusion
from argparse import ArgumentParser, Namespace


class EvalObject:
    def __init__(self, args):
        config_path = 'checkpoint/adaface_vit_base_kprpe_webface4m'
        self.train_set = FER(args,'checkpoint/quality',train=True)
        self.valid_set = FER(args,'checkpoint/quality',train=False)
        self.train_loader = DataLoader(self.train_set, batch_size=128, shuffle=False)
        self.valid_loader = DataLoader(self.valid_set, batch_size=128, shuffle=False)
        self.device = torch.device('cuda')
        self.model = kprpe_fer(config_path,cos=True)
        self.model.load_from_state_dict(args.ckpt_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.aligner = get_aligner(config_path).to(self.device)
        self.train_len = len(self.train_set)
        self.valid_len = len(self.valid_set)

    @torch.no_grad()
    def get_predictions(self, train=False):
        '''
        :param train: mode
        :return: accuracy, predictions, labels
        '''
        n = self.train_len if train else self.valid_len
        loader = self.train_loader if train else self.valid_loader
        y_hats, ys = [], []
        logits = []
        result = 0
        for img, label, q, c_q in tqdm(loader):
            img = img.to(self.device)
            _, ldmk, _, _, _, _ = self.aligner(img)
            label = label.to(self.device)
            embed, pred, _ = self.model(img, ldmk)
            logits.append(pred.cpu())
            y_hat = torch.argmax(pred, dim=-1).cpu().reshape(-1).tolist()
            y_hats += y_hat
            ys += label.cpu().reshape(-1).tolist()
            bs = label.shape[0]
            acc = get_acc(pred, label)
            result += (bs / n) * acc
            norms = embed.norm(p=2, dim=-1, keepdim=True)
        logits = torch.concat(logits,dim=0)
        return result, y_hats, ys, logits
