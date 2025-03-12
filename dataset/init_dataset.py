import sys
sys.path.append('..')
from dataset import FER
from argparse import Namespace

if __name__ == '__main__':
    args = {
        'dataset_path':'../data/AffectNet7',
        'world_size':1,

    }
    args = Namespace(**args)
    dataset = FER(args,ckpt_path='checkpoint/quality',train=True)
    dataset = FER(args,ckpt_path='checkpoint/quality',train=False)