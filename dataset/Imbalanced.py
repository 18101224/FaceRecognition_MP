import torchvision
import torchvision.transforms as T
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from datasets import load_dataset
from .transform import get_imagenet_transforms
from .augmentations import CIFAR10Policy
import os
import json


def get_cifar_dataset(dataset_name:str, root:str, imb_type:str, imb_factor:float, rand_number=0, train=True, transform=None, target_transform=None, download=True,):
    class CIFAR(torchvision.datasets.CIFAR10):
        def __init__(self, root, train, transform=None, target_transform=None, download=True, imb_type=None, imb_factor=None, rand_number=None):
            if dataset_name == 'cifar10':
                self.cls_num=10
            else:
                self.base_folder = 'cifar-100-python'
                self.url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
                self.filename = "cifar-100-python.tar.gz"
                self.tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
                self.train_list = [
                    ['train', '16019d7e3df5f24257cddd939b257f8d'],
                ]

                self.test_list = [
                    ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
                ]
                self.meta = {
                    'filename': 'meta',
                    'key': 'fine_label_names',
                    'md5': '7973b15100ade9c7d40fb424638fde48',
                }
                self.cls_num = 100
            super(CIFAR, self).__init__(root, train, transform, target_transform, download)
            np.random.seed(rand_number)
            if not train : 
                imb_factor = 1
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            self.gen_imbalanced_data(img_num_list)
            self.labels = self.targets
            self.img_num_list = self.get_cls_num_list()
        def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
            img_max = len(self.data)/cls_num
            img_num_per_lcs = []
            if imb_type == 'exp':
                for cls_idx in range(cls_num):
                    num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                    img_num_per_lcs.append(int(num))
            elif imb_type == 'step':
                for cls_idx in range(cls_num // 2):
                    img_num_per_lcs.append(int(img_max))
                for cls_idx in range(cls_num // 2):
                    img_num_per_lcs.append(int(img_max * imb_factor))
            else:
                img_num_per_lcs.extend([int(img_max)] * cls_num)
            return img_num_per_lcs
        
        def gen_imbalanced_data(self, img_num_per_cls): 
            new_data = []
            new_targets = []
            targets_np = np.array(self.targets, dtype=np.int64)
            classes = np.unique(targets_np)
            classes = np.concatenate([classes[1:], classes[:1]], axis=0)
            self.num_per_cls_dict = dict()
            for the_class, the_img_num in zip(classes, img_num_per_cls):
                self.num_per_cls_dict[the_class] = the_img_num
                idx = np.where(targets_np == the_class)[0]
                np.random.shuffle(idx)
                selec_idx = idx[:the_img_num]
                new_data.append(self.data[selec_idx, ...])
                new_targets.extend([the_class, ] * the_img_num)
            new_data = np.vstack(new_data)
            self.data = new_data
            self.targets = new_targets
        
        def get_cls_num_list(self):
            cls_num_list = []
            for i in range(self.cls_num):
                cls_num_list.append(self.num_per_cls_dict[i])
            return cls_num_list
        
        def __getitem__(self,idx):
            img, target = self.data[idx], int(self.targets[idx])
            img = Image.fromarray(img)
            
            if self.transform is not None : 
                if isinstance(self.transform, list):
                    samples = [tr(img) for tr in self.transform]
                else:
                    samples = self.transform(img)
            
            if self.target_transform is not None : 
                target = self.target_transform(target)

            return samples, target
        
        def __len__(self):
            return len(self.data)
        
    return CIFAR(root=root, train=train, transform=transform, target_transform=target_transform, download=download, imb_type=imb_type, imb_factor=imb_factor, rand_number=rand_number)



class Large_dataset(Dataset):
    def __init__(self, root, train, transform=None, use_randaug=False):
        super().__init__()
        self.img_path = []
        self.labels = []
        self.transform = transform 
        self.use_randaug = use_randaug
        self.dataset_name='inat' if 'inat' in root else 'imagenet_lt'
        if self.dataset_name == 'imagenet_lt':
            txt = 'ImageNet_LT_train.txt' if train else 'ImageNet_LT_test.txt' 

            with open(os.path.join(root, txt)) as f:
                for line in f:
                    img_path, label = line.split()
                    self.img_path.append(os.path.join(root, img_path.strip()))
                    self.labels.append(int(label.strip()))
            self.targets = self.labels 
        else:
            post = 'train' if train else 'val'
            json_path = os.path.join(root, f'{post}2018.json')
            with open(json_path, 'r') as f:
                dataset_json = json.load(f)

            images = dataset_json.get('images', [])
            annotations = dataset_json.get('annotations', [])

            image_id_to_path = {}
            for image_item in images:
                file_name = image_item.get('file_name')
                image_id = image_item.get('id')
                if file_name is None or image_id is None:
                    continue
                image_id_to_path[image_id] = os.path.join(root, file_name)

            for annotation in annotations:
                image_id = annotation.get('image_id')
                category_id = annotation.get('category_id')
                if image_id in image_id_to_path and category_id is not None:
                    self.img_path.append(image_id_to_path[image_id])
                    self.labels.append(int(category_id))

            self.targets = self.labels 

        self.img_num_list = self.get_img_num_per_cls()

    def get_img_num_per_cls(self):
        from collections import Counter
        counter = Counter(self.labels)
        num_classes = max(counter.keys())+1
        result = [0]*num_classes
        for key, value in counter.items():
            result[key] = value
        return np.array(result)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        path = self.img_path[idx]
        label = self.labels[idx]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            if isinstance(self.transform, list):
                samples_list = [tr(sample) for tr in self.transform]
                return samples_list, label  # Unpack images as separate outputs
            else:
                sample = self.transform(sample)
        return sample, label 

    
