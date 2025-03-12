import torch
import dataset
from models import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import *
from torchvision.utils import save_image
import numpy as np
import pickle
import os
from argparse import ArgumentParser
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
import torch.distributed as dist
from opt import SAM
from copy import deepcopy
import wandb

def get_args():
    args = ArgumentParser()
    args.add_argument('--learning_rate',type=float)
    args.add_argument('--batch_size',type=int)
    args.add_argument('--n_epochs',type=int)
    args.add_argument('--world_size', default=1,type=int)
    args.add_argument('--env_init_path')
    args.add_argument('--dataset')
    args.add_argument('--local_rank')
    args.add_argument('--rank')
    args.add_argument('--ada_loss',default=False)
    args.add_argument('--init_epoch',type=int,default=False)
    args.add_argument('--gamma',default=False,type=float)
    args.add_argument('--gp',default=False,type=float)
    args.add_argument('--k', default=1, type=int)
    args.add_argument('--save_path',default=0, type=int)
    args.add_argument('--name')
    args.add_argument('--wandb_token')
    args.add_argument('--opt',choices=['Adam','AdamW'],default='AdamW')
    args = args.parse_args()

    if args.world_size > 1 :
        init_process_group('nccl')
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(args.local_rank)
        dist.barrier()
        print(f'{dist.get_rank()} ready')

    return args

class Trainer:
    def __init__(self,args):
        self.args = args

        # set atts
        self.args.batch_size = deepcopy(int(self.args.batch_size/args.world_size))
        self.device = args.local_rank if args.world_size > 1 else torch.device('cuda')
        self.disable = False if args.world_size == 1 else dist.get_rank()!=0
        self.epoch = 0
        # get dataset and loaders
        self.entire_set = dataset.raf(args.dataset,train=True)
        self.selected_set = dataset.raf(args.dataset, train=True)
        self.valid_set = dataset.raf(args.dataset, train=False)
        if args.world_size > 1 :
            self.entire_sampler = DistributedSampler(self.entire_set, shuffle=True)
            self.selected_sampler = DistributedSampler(self.selected_set, shuffle=True)
            self.valid_sampler = DistributedSampler(self.valid_set, shuffle=False)
            self.shuffle = False
        else:
            self.shuffle = True
            self.entire_sampler = None
            self.selected_sampler = None
            self.valid_sampler = None
        self.entire_loader = DataLoader(self.entire_set,batch_size=self.args.batch_size,shuffle=self.shuffle,sampler=self.entire_sampler)
        self.selected_loader = DataLoader(self.selected_set,batch_size=self.args.batch_size,shuffle=self.shuffle,sampler=self.selected_sampler)
        self.valid_loader = DataLoader(self.valid_set,batch_size=self.args.batch_size,shuffle=False,sampler=self.valid_sampler)
        self.samplers = [self.entire_sampler, self.selected_sampler]

        # init models and opts
        self.classifier = get_ir().cuda()
        self.classifier.load_shadow()
        self.classifier.load_state_dict(torch.load(args.env_init_path,map_location=torch.device('cpu')))
        self.classifier = self.classifier.cuda()
        self.selector = Policy()
        self.selector = self.selector.cuda()
        if args.world_size > 1 :
            self.classifier = DistributedDataParallel(self.classifier,device_ids=[args.local_rank],find_unused_parameters=True)
            self.selector = DistributedDataParallel(self.selector,device_ids=[args.local_rank])
        print('model_loaded')
        self.c_opt = SAM(params=self.classifier.parameters(),base_optimizer=torch.optim.AdamW,lr=self.args.learning_rate,weight_decay=self.args.learning_rate/10)
        self.s_opt = torch.optim.Adam(self.selector.parameters(),lr=self.args.learning_rate)

        # init log and metric values
        if self.args.world_size ==1 or self.args.rank == 0:
            init_wandb(args)
        self.best_reward = -1e10
        self.best_acc = -1e10
        self.log = {'train_acc':[],'acc':[], 'reward':[], 'norm':[], 'init_reward':[],'len_selected':[]}
        if args.world_size == 1 or args.rank == 0  and not os.path.exists('sample'):
            os.mkdir('sample')
        self.save_path = args.save_path
        if args.world_size == 1 or args.rank == 0  and not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.mem_log = []
        self.best_env = deepcopy(self.classifier)
        print(self.args.init_epoch)
        if self.args.init_epoch > 0 :
            for epoch in range(self.args.init_epoch):
                self.run_epoch(epoch,train_classifier=False)
            self.selector.module.load_state_dict(self.init_weight)


        mean = self.get_feature_mean().reshape(1,-1)
        self.validation_classifier(0,mean)
        self.best_acc = self.sync(self.valid_acc)
        print(f'initial acc : {self.best_acc:.4f}')
    @torch.no_grad()
    def get_feature_mean(self):
        result = torch.zeros((256)).to(self.device)
        n=len(self.selected_set)
        for img, l,_ in tqdm(self.selected_loader,desc='get feature mean',disable=self.disable) :
            img = img.to(self.device)
            feature, _ = self.best_env(img)
            feature = torch.nn.functional.avg_pool2d(feature,kernel_size=14)
            feature = feature.reshape(-1,256)
            mean = torch.mean(feature,dim=0,keepdim=False)
            bs = img.shape[0]
            result+= mean*(bs/n)

        del img, l
        if self.args.world_size > 1 :
            dist.all_reduce(result,op=dist.ReduceOp.SUM)
            dist.barrier()
        return result

    @torch.no_grad()
    def sync_indices(self, indices):
        indices = torch.concat(indices, dim=0).float()
        local_size = torch.tensor([indices.size(0)],device=self.device)
        size_list = [torch.zeros_like(local_size) for _ in range(self.args.world_size)]
        dist.all_gather(size_list, local_size)
        size_list = [int(size.item()) for size in size_list]
        max_size = max(size_list)
        padding_length = max_size - indices.size(0)
        if padding_length > 0 :
            padding = torch.zeros(padding_length, 1, device=self.device)
            padding_tensor = torch.cat([indices, padding],dim=0)
        else :
            padding_tensor = indices

        gathered_tensors = [torch.zeros_like(padding_tensor) for _ in range(self.args.world_size)]
        dist.all_gather(gathered_tensors, padding_tensor)

        result = []
        for i, tensor in enumerate(gathered_tensors):
            original_size = size_list[i]
            if original_size > 0 :
                result.append(tensor[:original_size])

        result = torch.cat(result,dim=0).int().reshape(-1).tolist()

        return result
    def selection(self,tensor=False):
        imgs = []
        indices = []
        with torch.no_grad():
            mean = self.get_feature_mean() # ( 1, 256 )
            self.delete()
            mean = mean.unsqueeze(0)
            for img, label, idx in tqdm(self.entire_loader,desc='data selection',disable=self.disable):
                img = img.to(self.device)
                z,_ = self.classifier(img)
                idx = idx.to(self.device)
                bs = img.shape[0]
                condition = mean.repeat(bs,1)
                z = torch.nn.functional.avg_pool2d(z,kernel_size=14).reshape(-1,256)
                z = torch.concat((z,condition),dim=1)
                a = self.selector(z)
                # 결정한것만 넣는 것으로
                a = a>0.3
                a = a.reshape(-1,1).detach()
                idx = idx.reshape(-1,1)
                indices.append(idx[a.detach()].reshape(-1,1))
        if self.args.world_size > 1:
            indices = self.sync_indices(indices)
            dist.barrier()

        if len(indices) < 10 :
            return self.selected_set, mean
        if tensor :
            return imgs, mean
        else:
            return dataset.selected(self.args.dataset,indices), mean


    def get_reward(self,logit,label):
        mean = torch.mean(logit,dim=-1,keepdim=False)
        row = torch.arange(label.shape[0])
        true = logit[row.long(),label.long()]
        return true-mean

    def save_selected(self):
        imgs = self.selection(tensor=True)
        torch.save(imgs,'img_tensor.pth')

    def delete(self):
        del self.selected_loader
        del self.selected_sampler
        del self.selected_set
        self.samplers.pop(-1)

    @torch.no_grad()
    def sync(self, metric):
        metric = torch.tensor(metric, device = self.device)
        dist.all_reduce(metric, op=dist.ReduceOp.SUM)
        return metric.item()

    def sync_weight(self):
        for param in self.classifier.module.parameters():
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= self.args.world_size
    def acc(self,x,y,n):
        scalar = get_acc(x,y)
        bs = x.shape[0]
        return (bs/n)*scalar

    def train_classifier(self,epoch):
        loader,n = (self.entire_loader, len(self.entire_set)) if self.args.ada_loss else (self.selected_loader, len(self.selected_set))
        for img,label,_ in tqdm(loader,desc=f'epoch : {epoch} training_classifier', disable=self.disable):
            self.classifier.zero_grad()
            img = img.to(self.device)
            label = label.to(self.device)
            _, logit = self.classifier(img)
            gp = self.args.gp*get_grad_norm_x(img,self.classifier) if self.args.gp else 0
            loss = torch.nn.functional.cross_entropy(logit,label.long()) + gp
            loss.backward()
            self.train_acc += self.acc(logit,label,n)
            self.c_opt.first_step(zero_grad=True)
            self.classifier.zero_grad()
            _, logit = self.classifier(img)
            gp = self.args.gp*get_grad_norm_x(img,self.classifier) if self.args.gp else 0
            loss = torch.nn.functional.cross_entropy(logit,label.long()) + gp
            loss.backward()
            self.c_opt.second_step(zero_grad=True)
            if self.args.gamma :
                self.classifier.module.update(self.args.gamma)
                if self.args.world_size > 1 :
                    self.sync_weight()
        if self.args.gamma :
            self.classifier.module.apply_ema()

    def train_selector(self,epoch,mean):
        loader = self.entire_loader if self.args.ada_loss else self.selected_loader
        for img,label, _ in tqdm(loader,desc=f'epoch : {epoch} training selector', disable=self.disable):
            self.s_opt.zero_grad()
            img = img.to(self.device)
            label = label.to(self.device)
            with torch.no_grad():
                z, logit = self.best_env(img)
            reward = self.get_reward(logit, label)
            bs = img.shape[0]
            c = mean.repeat(bs,1)
            z = torch.nn.functional.avg_pool2d(z,kernel_size=(14,14)).reshape(-1,256)
            z = torch.concat((z,c),dim=1)
            a = self.selector(z)
            loss = torch.mean(-(torch.log(a)).reshape(-1)*reward,dim=0,keepdim=False)
            loss.backward()
            self.s_opt.step()

    @torch.no_grad()
    def validation_classifier(self,epoch,mean):
        len_selected = 0
        for img, label, _ in tqdm(self.valid_loader, desc=f'epoch {epoch} validation, traini_acc:{self.train_acc:.4f}',
                                  disable=self.disable):
            img = img.to(self.device)
            label = label.to(self.device)
            z, logit = self.classifier(img)
            reward = self.get_reward(logit, label).to(self.device)
            bs = label.shape[0]
            condition = mean.repeat(bs, 1)
            z = torch.nn.functional.avg_pool2d(z, kernel_size=(14, 14)).reshape(-1, 256)
            z = torch.concat((z, condition), dim=1)
            a = self.selector(z).reshape(-1)
            a = a > 0.3
            self.reward += torch.mean(reward[a], dim=0).reshape(1).item() * len(a)
            self.valid_acc += self.acc(logit, label, len(self.valid_set))
        return len_selected

    def run_epoch(self,epoch,train_classifier=True):

        # data selection
        self.selector.eval()
        self.classifier.eval()

        self.selected_set, mean = self.selection()
        self.selected_sampler = DistributedSampler(self.selected_set,shuffle=True) if self.args.world_size > 1 else None
        self.selected_loader = DataLoader(self.selected_set,batch_size=self.args.batch_size,shuffle=False,sampler=self.selected_sampler)
        self.samplers.append(self.selected_sampler)

        # training classifier
        self.classifier.train()
        self.train_acc = 0

        if train_classifier:
            self.train_classifier(epoch)
            self.train_acc = self.sync(self.train_acc)

        # training selector
        self.classifier.eval()

        self.selector.train()
        mean = mean.reshape(1,-1)
        for _ in range(self.args.k):
            self.train_selector(epoch,mean)

        # validation
        self.selector.eval()
        self.valid_acc = 0
        self.reward = 0
        len_selected = self.validation_classifier(epoch,mean)


        self.valid_acc = self.sync(self.valid_acc)
        self.reward = self.sync(self.reward)
        len_selected = self.sync(len_selected)
        self.reward = self.reward/len_selected if len_selected > 0 else -1
        dist.barrier()

        if self.valid_acc > self.best_acc :
            self.best_env = deepcopy(self.classifier)
        else:
            self.classifier = deepcopy(self.best_env)

        if train_classifier:
            if self.args.world_size == 1 or self.args.rank == 0:
                self.log['train_acc'].append(self.train_acc)
                self.log['acc'].append(self.valid_acc)
                self.log['reward'].append(self.reward)
                self.log['len_selected'].append(len(self.selected_set))
                wandb.log(
                    {'epoch':epoch,
                     'train_acc':self.train_acc,
                     'valid_reward':self.reward,
                     'valid_acc':self.valid_acc,
                     'selected_len':len(self.selected_set)}
                )
                if self.best_acc < self.valid_acc :
                    self.best_acc = self.valid_acc
                    if self.args.world_size > 1:
                        torch.save(self.classifier.module.state_dict(),f'{self.save_path}/best_acc_c.pth')
                        torch.save(self.selector.module.state_dict(), f'{self.save_path}/best_acc_s.pth')
                    else:
                        torch.save(self.classifier.state_dict(),f'{self.save_path}/best_acc_c.pth')
                        torch.save(self.selector.state_dict(), f'{self.save_path}/best_acc_s.pth')
                    torch.save(mean, f'{self.save_path}/best_acc_ct.pth')
                if self.best_reward < self.reward :
                    self.best_reward = self.reward
                    if self.args.world_size > 1:
                        torch.save(self.classifier.module.state_dict(), f'{self.save_path}/best_reward_c.pth')
                        torch.save(self.selector.module.state_dict(),f'{self.save_path}/best_reward_s.pth')
                    else:
                        torch.save(self.classifier.state_dict(), f'{self.save_path}/best_reward_c.pth')
                        torch.save(self.selector.state_dict(),f'{self.save_path}/best_reward_s.pth')
                    torch.save(mean,f'{self.save_path}/best_reward_ct.pth')
        else:
            self.log['init_reward'].append(self.reward)
            if self.best_reward < self.reward:
                self.init_weight = self.selector.module.state_dict()

        if self.args.world_size > 1:
            dist.barrier()

    def train(self,epochs):

        for epoch in range(epochs):
            self.epoch = epoch
            if self.args.world_size > 1 :
                for sampler in self.samplers :
                    sampler.set_epoch(epoch)
            self.run_epoch(epoch)
            if self.args.world_size == 1 or self.args.rank == 0 :
                avail, occ = get_mem()
                self.mem_log.append((avail,occ))
                with open(f'{self.save_path}/mem_log.pkl','wb') as f :
                    pickle.dump(self.mem_log,f)
                with open(f'{self.save_path}/log.pkl','wb') as f:
                    pickle.dump(self.log,f)

        return self.best_acc, self.best_reward
if __name__ == '__main__':
    args = get_args()
    trainer = Trainer(args)
    gamma = [args.gamma]
    ks = [1]
    gps = [False]
    idx = args.save_path
    log = get_dict('rl_log.pkl')
    for g in gamma :
        for k in ks :
            for gp in gps :
                args.gp=gp
                args.gamma = g
                args.k= k
                args.save_path = f'RL{idx}'
                trainer = Trainer(args)
                best_acc, best_reward =  trainer.train(args.n_epochs)

                log['best_reward'].append(best_reward)
                log['best_acc'].append(best_acc)
                log['gamma'].append(g)
                log['k'].append(k)
                log['gp'].append(gp)
                log['save path'].append(args.save_path)
                idx+=1
                del trainer

    save_dict(log,'rl_log.pkl')