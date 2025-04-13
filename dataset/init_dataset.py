
from fer import FER
from argparse import Namespace

if __name__ == '__main__':
    args = {
        'dataset_path':'../data/RAF-DB',
        'world_size':1,

    }
    args = Namespace(**args)
    dataset = FER(args,ckpt_path='checkpoint/quality',train=True)
    dataset = FER(args,ckpt_path='checkpoint/quality',train=False)