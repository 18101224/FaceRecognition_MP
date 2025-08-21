import wandb,os,torch
import numpy as np 
from dataset import get_cifar_dataset, ImbalanceSVHN, DistributedSamplerWrapper, ImbalancedDatasetSampler, get_cl_transforms
from models import resnet32, CosClassifier, SoftMaxClassifier, get_imgnet_resnet
from argparse import ArgumentParser
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from Loss import BalSCL
from utils import get_exp_id
from tqdm import tqdm 



def get_dataset(args, train,transform):
    dataset_name = args.dataset_name
    if 'cifar' in dataset_name:
        n_c = 100 if '100' in args.dataset_name else 10
        result =  get_cifar_dataset(root=args.dataset_path, num_classes=n_c,  im_factor=args.imb_factor,mode=train, imb_type=args.imb_type, indexes=None, use_kfold=args.kfold, transform=transform)
    return result 

def get_loaders(args, train_set, valid_set):
    if args.world_size > 1: 
        if args.use_sampler : 
            sampler = DistributedSamplerWrapper(sampler = ImbalancedDatasetSampler(train_set, labels=train_set.labels,shuffle=True))
        else:
            sampler = DistributedSampler(train_set)
        valid_sampler = DistributedSampler(valid_set)
    else:
        if args.use_sampler : 
            sampler = ImbalancedDatasetSampler(train_set, labels=train_set.labels,shuffle=True)
        else:
            sampler = None
        valid_sampler = None 
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, sampler=valid_sampler, num_workers=args.num_workers, pin_memory=True)
    return train_loader, valid_loader


def get_model(args):
    if 'cifar' in args.dataset_name:
        n_c = 100 if '100' in args.dataset_name else 10
        backbone = resnet32(num_class=1,use_norm=True)
        return CosClassifier(num_classes=n_c, backbone=backbone, dim=args.dim, num_hidden_layers=0)
    elif 'svhn' in args.dataset_name:
        bacbone = get_imgnet_resnet()
        model = CosClassifier(num_classes=10, backbone=bacbone, dim=2048, num_hidden_layers=4)
        return model


def get_loss(args, cls_num_list):
    if args.loss == "BCL":
        return BalSCL( cls_num_list=cls_num_list, temperature=args.temperature)
    
def get_args():
    args = ArgumentParser()
    args.add_argument('--learning_rate', type=float)
    args.add_argument('--batch_size', type=int)
    args.add_argument('--n_epochs', type=int)


    args.add_argument('--dim', type=int)
    args.add_argument('--temperature', type=float)


    args.add_argument('--rank')
    args.add_argument('--world_size', type=int, default=1)
    args.add_argument('--local_rank')
    args.add_argument('--num_workers', type=int, default=16)

    args.add_argument('--wandb_token')
    args.add_argument('--server')
    
    args.add_argument('--loss', choices=['BCL'])
    args.add_argument('--dataset_name', choices=['cifar100','cifar10'])
    args.add_argument('--dataset_path')
    args.add_argument('--kfold', type=float, default=False)
    args.add_argument('--use_sampler', default=False)

    args.add_argument('--imb_type', choices=['exp', 'step'], default='exp')
    args.add_argument('--imb_factor', type=int, default=100)
    args.add_argument('--cl_views', default=None, choices=['sim-sim', 'sim-rand', 'randstack-randstack'])
    args.add_argument('--randaug_n', type=int, default=2)
    args.add_argument('--randaug_m', type=int, default=15)
    
    args = args.parse_args()
    
    if args.world_size > 1: 
        args.rank = int(os.environ['RANK'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.batch_size = args.batch_size // args.world_size
        init_process_group('nccl')
        torch.cuda.set_device(args.local_rank)
    else:
        torch.cuda.set_device(0)
    return args


class Pretrainer:
    def __init__(self, args):
        self.args = args
        self.device = args.local_rank if args.world_size > 1 else torch.device('cuda')
        # dataset and loader
        if args.cl_views is not None:
            self.transforms = get_cl_transforms(args)
            self.train_dataset = get_dataset(args, train='train', transform=self.transforms)
        else:
            self.train_dataset = get_dataset(args, train='train', transform=None)
        self.valid_dataset = get_dataset(args, train='validation', transform=None)
        self.train_loader, self.valid_loader = get_loaders(args, self.train_dataset, self.valid_dataset)

        # loss  
        self.loss = get_loss(args, self.train_dataset.img_num_list)
        
        # model and opt 
        self.model = get_model(args)
        self.model.cuda()
        if args.world_size > 1:
            self.model = DDP(self.model, device_ids=[args.local_rank], find_unused_parameters=True)
        self.opt = torch.optim.SGD(self.model.parameters(), lr=args.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=args.n_epochs, eta_min=0)

        # log utils 
        if args.world_size == 1 or args.rank == 0 :
            self.id = get_exp_id(args)
            self.init_wandb()
            if not os.path.exists(f'checkpoint/{self.id}'):
                os.makedirs(f'checkpoint/{self.id}')
            self.save_dir = f'checkpoint/{self.id}'
        self.best_loss = float('inf')
        self.best_acc = -float('inf')


    def compute_prototypes(self):
        self.model.eval()
        features, labels = [], []
        for batch in tqdm(self.train_loader, desc='Computing Prototypes', disable=self.args.world_size>1 and self.args.rank!=0):
            if self.args.cl_views is not None:
                img,_,_,label = batch 
            else:
                img, label = batch
            img = img.to(self.device)
            label = label.to(self.device)
            feature, _ = self.model(img, features=True)
            features.append(feature)
            labels.append(label)
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
        cls_counts = torch.eye(len(self.train_dataset.img_num_list),device=self.device)[labels].sum(dim=0).squeeze().to(features.device).reshape(-1,1)
        prototypes = []
        for i in range(len(self.train_dataset.img_num_list)):
            prototypes.append(features[labels==i].mean(dim=0).reshape(-1))
        prototypes = torch.nn.functional.normalize(torch.stack(prototypes).reshape(-1,features.shape[-1])/cls_counts, p=2, dim=1)
        return prototypes

    def few_shot_inference(self, prototypes, features):
        logits = features @ prototypes.T
        return logits, torch.argmax(logits, dim=1)
    
    def init_wandb(self):
        with open(self.args.wandb_token, 'r') as f:
            token = f.readline().strip()
        wandb.login(key=token)
        wandb.init(
            project=f'{self.args.dataset_name} Pre-Training',
            name=self.id,
            config=dict(vars(self.args), model='resnet32')  # This will include all existing arguments
        )
    
    def save(self):
        to_save = self.model if self.args.world_size == 1 else self.model.module 
        torch.save(to_save.state_dict(), f'{self.save_dir}/best_loss.pth')

    def sync_tensor(self, tensor):
        if tensor is not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor, device=self.device)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOP.SUM)
        return tensor.item().detach().cpu()
    
    def run_train_epoch(self):
        total_loss = 0 

        for batch in tqdm(self.train_loader, desc='Training', disable=self.args.world_size>1 and self.args.rank!=0):
            if self.args.cl_views is not None:
                _, images1, images2, labels = batch 
                images = torch.cat([images1, images2], dim=0)
            else:
                images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.opt.zero_grad()
            kernel = self.model.get_kernel() if self.args.world_size==1 else self.model.module.get_kernel()
            features, _  = self.model(images, features=True)
            f2, f3 = torch.split(features, (labels.shape[0], labels.shape[0]), dim=0)
            features = torch.cat([f2.unsqueeze(1),f3.unsqueeze(1)],dim=1)
            loss = self.loss.compute_loss(centers1=kernel.T, features=features, targets=labels)
            loss.backward()
            self.opt.step()
            total_loss += loss.item()*labels.shape[0]
        
        if self.args.world_size > 1 :
            total_loss = self.sync_tensor(total_loss)
        return total_loss / len(self.train_dataset)
    
    @torch.no_grad()
    def run_valid_epoch(self):
        accs = 0 
        prototypes = self.compute_prototypes()
        for batch in self.valid_loader:
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            features, _ = self.model(images, features=True)
            logits, preds = self.few_shot_inference(prototypes, features)
            accs += (preds == labels).sum().float().item()
        if self.args.world_size > 1 :
            accs = self.sync_tensor(accs)
        return accs / len(self.valid_dataset)
    
    def run_epoch(self):
        self.model.train()
        train_loss = self.run_train_epoch()
        self.model.eval()
        valid_acc = self.run_valid_epoch()
        self.scheduler.step()
        if valid_acc > self.best_acc : 
            self.best_acc = valid_acc
            self.save()
        if self.args.world_size == 1 or self.args.rank == 0 : 
            wandb.log({
                'train_loss': train_loss,
                'valid_acc': valid_acc,
                'best_acc': self.best_acc,
            })

    def train(self):
        for epoch in range(self.args.n_epochs):
            if self.args.world_size > 1:
                self.train_loader.sampler.set_epoch(epoch)
                self.valid_loader.sampler.set_epoch(epoch)
            self.run_epoch()

if __name__ == '__main__':
    args = get_args()
    pretrainer = Pretrainer(args)
    pretrainer.train()