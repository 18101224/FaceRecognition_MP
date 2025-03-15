from models import kprpe_fer
import torch
from analysis import compute_all_label_noise, plot_j_distribution
from aligners import get_aligner
from dataset import FER
from torch.utils.data import DataLoader
from argparse import Namespace

if __name__ == '__main__':
    device = torch.device('cuda')
    gen_model = kprpe_fer('checkpoint/vanilla12m', cos=False)
    gen_model.load_from_state_dict('checkpoint/vanilla12m')
    gen_model = gen_model.to(device)
    aligner = get_aligner('checkpoint/adaface_vit_base_kprpe_webface12m')  # Replace with actual path to your aligner configuration
    aligner = aligner.to(device)  # Replace with actual aligner checkpoint path
    aligner = aligner.to(device)
    args = Namespace(**{'dataset_path':'../data/AffectNet7', 'world_size':1})
    dataset = FER(args, 'checkpoint/quality', train=True)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    # gen_model.eval()
    # info = compute_all_label_noise(dataloader,gen_model, aligner, device)
    # plot_j_distribution(info,save_path='results/eval_j_dist.png')
    gen_model.train()
    info = compute_all_label_noise(dataloader,gen_model, aligner, device)
    plot_j_distribution(info, save_path='results/train_j_dist.png')

