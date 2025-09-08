import torch 
from tqdm import tqdm 
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from dataset import ClassBatchSampler, FER, ImbalancedDatasetSampler, DistributedSamplerWrapper, get_transform
from models.modules import get_QCS_model
from models import ImbalancedModel
import wandb , os
from argparse import ArgumentParser
from opt import SAM 
import torch.distributed as dist
from torch.distributed import init_process_group
from datetime import timedelta
from utils import get_exp_name
import numpy as np
from utils import get_acc

def get_args():
    args = ArgumentParser()
    args.add_argument('--world_size',type=int,default=1)
    args.add_argument('--local_rank',type=int,default=None)
    args.add_argument('--rank',type=int,default=None)
    args.add_argument('--batch_size',type=int,default=128)
    args.add_argument('--n_epochs',type=int,default=200)
    args.add_argument('--learning_rate',type=float,default=0.0001)
    args.add_argument('--weight_decay',type=float,default=1e-4)
    args.add_argument('--dim',type=int,default=768)
    args.add_argument('--guide_path',)
    args.add_argument('--dataset_path',)
    args.add_argument('--model_type',type=str,default='ir50')
    args.add_argument('--dataset_name',type=str,default='RAF-DB')
    args.add_argument('--k',type=int,default=2)
    args.add_argument('--use_sampler',default=False)
    args.add_argument('--num_workers',type=int,default=0)
    args = args.parse_args()
    if args.world_size > 1 :
        init_process_group('nccl',world_size=args.world_size,
                           timeout=timedelta(minutes=60))
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.rank = int(os.environ['RANK'])
        args.batch_size = args.batch_size // args.world_size
        torch.cuda.set_device(args.local_rank)
    else:
        torch.cuda.set_device(0)
    return args

class Trainer:
    def __init__(self,args):
        self.args = args 
        self.device = self.args.local_rank if self.args.world_size > 1 else torch.device('cuda')
        self.train_set = FER(args,train=True,idx=False, transform=get_transform(self.args,train=True))
        self.valid_set = FER(args,train=False,idx=False, transform=get_transform(self.args,train=False))
        if args.use_sampler : 
            train_sampler = ImbalancedDatasetSampler(self.train_set,labels=self.train_set.labels) if self.args.world_size == 1 \
                else DistributedSamplerWrapper(ImbalancedDatasetSampler(self.train_set,labels=self.train_set.labels),shuffle=True)
        else:
            train_sampler = None if self.args.world_size == 1 else DistributedSampler(self.train_set,shuffle=True)
        self.valid_sampler = DistributedSampler(self.valid_set,shuffle=False) if self.args.world_size > 1 else None
        self.train_loader = DataLoader(self.train_set,batch_size=self.args.batch_size,sampler=train_sampler)
        self.valid_loader = DataLoader(self.valid_set,batch_size=self.args.batch_size,sampler=self.valid_sampler)
        self.fetcher = ClassBatchSampler(args, transform=get_transform(args,train=True),idx=False)
        self.model = get_QCS_model(self.args.model_type,self.args.dim,7).cuda()
        self.model = self.model.to(self.device) if self.args.world_size == 1 else DDP(self.model,device_ids=[self.args.local_rank],find_unused_parameters=True)
        self.opt = SAM(self.model.parameters(),base_optimizer=torch.optim.AdamW,lr=self.args.learning_rate,weight_decay=self.args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt,gamma=0.98)
        self.guide_network = ImbalancedModel(num_classes=7,model_type='ir50',)
        self.guide_network.load_state_dict(torch.load(f'{self.args.guide_path}/latest.pth',weights_only=False)['model_state_dict'])
        self.guide_network.cuda()
        self.id = self.init_wandb()
        self.init_sims()
    
    def init_sims(self):
        # dynamic or static 
        weight = self.guide_network.get_kernel() # dim, num_classes
        sims = (weight.T @ weight ).clone()
        sims.fill_diagonal_(-np.inf)
        sorted_classes = sims.sort(dim=-1,descending=True)[1]
        hard_indices = sorted_classes[:,0].reshape(-1)
        cls_counts = torch.tensor(self.train_set.get_img_num_per_cls()).sort(dim=-1,descending=True)[-1].reshape(-1)
        bank = cls_counts.unsqueeze(0).repeat(7,1)
        bank[:,0] = hard_indices
        for i in range(7):
            if bank[i,0] == bank[i,1]:
                bank[i,1] = cls_counts[2] if not i==2 else cls_counts[3]
        self.bank=bank.to(self.device)

    def init_wandb(self,):
        wandb.login()
        exp_name = get_exp_name()
        os.makedirs(f'checkpoint/{exp_name}',exist_ok=True)
        self.save_dir = f'checkpoint/{exp_name}'
        wandb.init(project='QCS_hcm-'+self.args.dataset_name,name=exp_name,config=self.args)
        self.best_acc = 0
        return exp_name

    def run_train_forward(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        positive, y_p = self.fetcher.sample_pairs(labels,k=1,num_workers=self.args.num_workers)
        hard_neg, y_n1 = self.fetcher.sample_pairs(self.bank[labels,0],k=self.args.k,num_workers=self.args.num_workers)
        head_neg, y_n2 = self.fetcher.sample_pairs(self.bank[labels,1],k=self.args.k,num_workers=self.args.num_workers) #k,bs 
        outputs = self.model(images, positive, hard_neg, head_neg)  
        outputs = torch.cat(outputs,dim=0)
        label = torch.cat([labels.reshape(-1,1),y_p.reshape(-1,1),y_n1.reshape(-1,1),y_n2.reshape(-1,1)]*2,dim=0).squeeze(-1)
        loss = torch.nn.functional.cross_entropy(outputs,label)
        return loss, outputs[:images.shape[0]]

    @torch.no_grad()
    def run_valid_forward(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        logits = self.model(images)
        loss = torch.nn.functional.cross_entropy(logits,labels)
        return loss, logits
    
    def train_epoch(self,):
        self.model.train()
        total_loss = 0
        total_acc = 0
        for images, labels in tqdm(self.train_loader, disable=self.args.world_size > 1 and self.args.rank != 0):
            loss, outputs = self.run_train_forward(images[0], labels)
            loss.backward()
            self.opt.first_step(zero_grad=True)
            loss, outputs = self.run_train_forward(images[0], labels)
            loss.backward()
            self.opt.second_step(zero_grad=True)
            with torch.no_grad():
                total_loss += loss.detach().cpu().item()*(labels.shape[0])
                total_acc += get_acc(outputs.cpu(), labels)*labels.shape[0]
        
        if self.args.world_size > 1 :
            total_loss = dist.all_reduce(torch.tensor([total_loss]),op=dist.ReduceOp.SUM).item()
            total_acc = dist.all_reduce(torch.tensor([total_acc]),op=dist.ReduceOp.SUM).item()
        
        total_loss = total_loss/len(self.train_set)
        total_acc = total_acc/len(self.train_set)

        return total_loss, total_acc

    def valid_epoch(self,):
        self.model.eval()
        total_loss = 0
        total_acc = 0
        for images, labels in tqdm(self.valid_loader, disable=self.args.world_size > 1 and self.args.rank != 0):
            loss, outputs = self.run_valid_forward(images, labels)
            total_loss += loss.detach().cpu().item()*(labels.shape[0])
            total_acc += get_acc(outputs.cpu(), labels)*labels.shape[0]
        if self.args.world_size > 1 :
            total_loss = dist.all_reduce(torch.tensor([total_loss]),op=dist.ReduceOp.SUM).item()
            total_acc = dist.all_reduce(torch.tensor([total_acc]),op=dist.ReduceOp.SUM).item()
        total_loss = total_loss/len(self.valid_set)
        total_acc = total_acc/len(self.valid_set)
        return total_loss, total_acc

    def save(self,is_best=False):
        to_save = self.model if self.args.world_size == 1 else self.model.module
        if is_best:
            torch.save(to_save.state_dict(),f'{self.save_dir}/best_acc.pth')
        else:
            torch.save(to_save.state_dict(),f'{self.save_dir}/latest.pth')

    def run_epoch(self):
        train_loss, train_acc = self.train_epoch()
        valid_loss, valid_acc = self.valid_epoch()

        if self.best < valid_acc : 
            self.best_acc = valid_acc
            if self.args.world_size == 1 or self.args.rank == 0 :
                self.save(True)
        else:
            if self.args.world_size == 1 or self.args.rank == 0 :
                self.save(False)

        log = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'valid_loss': valid_loss,
            'valid_acc': valid_acc,
            'best_acc': self.best_acc,
            'epoch': self.epoch
        }
        wandb.log(log)
        return train_loss, train_acc, valid_loss, valid_acc

    def train(self,):
        for epoch in range(self.args.n_epochs):
            self.epoch = epoch 
            if self.args.world_size > 1:
                self.train_loader.sampler.set_epoch(epoch)
                self.valid_loader.sampler.set_epoch(epoch)
            self.run_epoch()
        
        if self.args.world_size == 1 or self.args.rank == 0 :
            import pickle 
            with open(f'{self.save_dir}/log.pkl','wb') as f:
                pickle.dump(self.log,f)



if __name__ == '__main__':
    args = get_args()
    trainer = Trainer(args)
    trainer.train()
