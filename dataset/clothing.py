from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import os, torch
from PIL import Image

class Clothing1m(Dataset):
    def __init__(self, root, train=True):
        super().__init__()
        self.root = root
        post = 'clothing1m' if train else 'clothing10k_test'
        path = os.path.join(self.root,post+'.npz')
        data = np.load(path)
        self.images = data['arr_0']
        self.labels = torch.tensor(data['arr_1']).long()
        self.transform =  transforms.Compose([
            transforms.Resize((112,112)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.6959, 0.6537, 0.6371], std=[0.3113, 0.3192,0.3214])
        ]) if train else transforms.Compose([
            transforms.Resize((112,112))
            ,transforms.ToTensor(),
            transforms.Normalize(mean=[0.6959, 0.6537, 0.6371],std=[0.3113, 0.3192,0.3214])
        ])

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, self.labels[idx]

    def get_labels(self):
        return self.labels