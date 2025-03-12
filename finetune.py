import torch
from torch import distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchsampler import ImbalancedDatasetSampler
from catalyst.data import DistributedSamplerWrapper
import os
import pickle
from dataset import *
from models import *
from torch.nn.parallel import DistributedDataParallel as DDP
from opt import SAM
from tqdm import tqdm
from utils import *
from argparse import ArgumentParser
import warnings
import pandas as pd
from copy import deepcopy

warnings.filterwarnings('ignore')

def get_args():
    args = ArgumentParser()
    args.add_argument('--data_path')
    args.add_argument('--learning_rate',type=float)
    args.add_argument('--ckpt_path')
    args.add_argument('--world_size',type=int,default=1)
    args.add_argument('--bs_list',type=int,nargs='+',required=True)
    args.add_argument('--lr_list',type=float,nargs='+',required=True)
    args.add_argument('--batch_size',type=int)
    args.add_argument('--local_rank')
    args.add_argument('--rank')
    args.add_argument('--n_epochs', type=int)
    args.add_argument('--idx',type=int)
    args.add_argument('--name')
    args.add_argument('--log_file')
    args.add_argument('--mode',choices=['r','a'])
    args = args.parse_args()

    if args.world_size > 1 :
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group('nccl')

    return args

# args.data_path
class Trainer:
    def __init__(self,args):
        self.args = args

        # get data utils
        self.train_set, self.valid_set, self.train_loader, self.valid_loader, self.train_sampler = \
        get_data(args)
        self.train_len, self.valid_len = len(self.train_set), len(self.valid_set)
        self.device = torch.device('cuda') if args.world_size == 1 else args.local_rank


        # get models
        self.classifier, self.selector, self.condition = load_ckpt(args.ckpt_path,args.mode)
        self.classifier = self.classifier.cuda()
        self.selector = self.selector.cuda()
        self.selector.eval()

        if self.args.world_size > 1 :
            self.classifier = DDP(self.classifier, device_ids=[self.device], find_unused_parameters=True)
            self.selector = DDP(self.selector, device_ids=[self.device],find_unused_parameters=True)

        #opts
        self.c_opt = SAM(self.classifier.parameters(),base_optimizer=torch.optim.Adam,lr=args.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.c_opt, T_max=args.n_epochs)

        # log utils
        self.best = -1e10
        self.log = {'train_acc':[], 'train_loss':[],'train_ada_loss':[], 'valid_acc':[], 'valid_loss':[],'valid_ada_loss':[]}
        self.save_path = os.path.join('.',str(self.args.idx))

        if (args.world_size==1 or args.rank==0) and not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.set_metrics_zero()
        self.train_acc = 0;
        self.train_loss = 0;
        self.train_ada_loss = 0
        self.valid_acc = 0;
        self.valid_loss = 0;
        self.valid_ada_loss = 0
        self.disable = self.args.rank != 0 and self.args.world_size!=1

        self.ratio = 0
    def set_metrics_zero(self):
        self.train_acc = 0 ; self.train_loss = 0 ; self.train_ada_loss = 0
        self.valid_acc = 0 ; self.valid_loss = 0 ; self.valid_ada_loss = 0

    def append_metrics(self):
        self.log['train_acc'].append(self.train_acc); self.log['valid_acc'].append(self.valid_acc)
        self.log['train_loss'].append(self.train_loss); self.log['valid_loss'].append(self.valid_loss)
        self.log['train_ada_loss'].append(self.train_ada_loss); self.log['valid_ada_loss'].append(self.valid_ada_loss)

    def sync(self, metric):
        metric_tensor = torch.tensor(metric, device=self.device)
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
        averaged_metric = metric_tensor.item() / self.args.world_size
        return averaged_metric
    @torch.no_grad()
    def get_ada_weight(self, z):
        z = torch.nn.functional.avg_pool2d(z,kernel_size=(14,14))
        z = z.reshape(z.shape[0],-1)
        condition = self.condition.reshape(1,-1).repeat(z.shape[0],1).to(self.device)
        x = torch.concat((z,condition),dim=-1)
        weight = self.selector(x)
        return weight
    @torch.no_grad()
    def calc_metrics(self, logit, labels,loss, ada_loss,train):
        bs = logit.shape[0]
        acc = get_acc(logit,labels)
        loss = loss.item()
        ada_loss=ada_loss.item()

        if train :
            self.train_acc+=(bs/self.train_len)*acc
            self.train_loss+=(bs/self.train_len)*loss
            self.train_ada_loss+=(bs/self.train_len)*ada_loss
            if self.args.world_size > 1:
                self.train_ada_loss = self.sync(self.train_ada_loss)
                self.train_loss = self.sync(self.train_loss)
                self.train_acc = self.sync(self.train_acc)
                dist.barrier()

        else:
            self.valid_acc+=(bs/self.valid_len)*acc ;
            self.valid_loss+=(bs/self.valid_len)*loss ;
            self.valid_ada_loss+=(bs/self.valid_len)*ada_loss;
            if self.args.world_size > 1:
                self.valid_ada_loss = self.sync(self.valid_ada_loss)
                self.valid_loss = self.sync(self.valid_loss)
                self.valid_acc = self.sync(self.valid_acc)
    def run_forward(self, imgs, labels):
        z, logit = self.classifier(imgs)
        weight = self.get_ada_weight(z).reshape(-1)
        loss = torch.nn.functional.cross_entropy(logit,labels,reduction='none')
        ada_loss = weight.reshape(-1)*loss
        ada_loss = torch.mean(ada_loss, dim=0,keepdim=False)
        with torch.no_grad():
            loss = torch.mean(loss,dim=0,keepdim=False)
        return loss, ada_loss , logit

    def save(self):
        if self.args.world_size > 1 :
            torch.save(self.classifier.module.state_dict(),f'{self.save_path}/classifier.pth')
        else:
            torch.save(self.classifier.state_dict(),f'{self.save_path}/classifier.pth')
    def run_epoch(self,epoch):
        if self.args.world_size > 1:
            self.train_sampler.set_epoch(epoch)

        self.classifier.train()
        for imgs, labels,_ in tqdm(self.train_loader, desc=f'training {epoch} : \
            training_acc : {self.train_acc:.4f} best :{self.best:.4f}', disable=self.disable):
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            loss, ada_loss, logit = self.run_forward(imgs,labels)
            ada_loss.backward()

            self.c_opt.first_step(zero_grad=True)
            self.classifier.zero_grad()
            self.calc_metrics(logit, labels, loss, ada_loss, train=True)



            loss, ada_loss, logit = self.run_forward(imgs,labels)
            ada_loss.backward()
            self.c_opt.second_step(zero_grad=True)

        self.classifier.eval()
        with torch.no_grad():
            for imgs,labels,_ in tqdm(self.valid_loader,  desc=f'validation {epoch} :\
             validation : {self.valid_acc:.4f} best : {self.best:.4f}',disable=self.disable):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                loss, ada_loss, logit = self.run_forward(imgs,labels)
                self.calc_metrics(logit,labels,loss,ada_loss,train=False)


        if self.args.world_size == 1 or self.args.rank == 0:
            if self.best < self.valid_acc  :
                self.save()
                self.best = deepcopy(self.valid_acc)

        self.scheduler.step()

    @torch.no_grad()
    def valid_alone(self):
        classifier = get_ir().load_state_dict(torch.load(f'{self.save_path}/classifier.pth'))
        classifier = classifier.to(self.device)
        valid_set = raf(self.args.data_path,train=False)
        valid_loader = DataLoader(valid_set,batch_size=512,shuffle=False)
        best = 0
        for img, label in valid_loader:
            img = img.to(self.device)
            label = label.to(self.device)
            _, logit = classifier(img)
            acc = get_acc(logit,label)
            best += (label.shape[0]/len(valid_set))*acc
        return best
    def train(self):
        for epoch in tqdm(range(self.args.n_epochs)):
            self.set_metrics_zero()
            self.run_epoch(epoch)
            self.append_metrics()
            if self.args.world_size == 1 or self.args.rank==0 :
                with open(f'{self.save_path}/log.pkl','wb') as f :
                    pickle.dump(self.log,f)


        return self.best, self.log, self.save_path


if __name__ == '__main__':
    args = get_args()
    log = get_pd(args.log_file)
    log = pd.DataFrame(log)
    idx = args.idx
    if 'Affect' in args.data_path :
        prefix = 'Affect_'
    else:
        prefix = 'RAF_'
    args.name = f'{args.name}_{args.mode}'
    if (args.rank == 0 or args.world_size == 1) and (not os.path.exists(args.name)):
        os.mkdir(args.name)
    for bs in args.bs_list :
        for lr in args.lr_list :
            args.batch_size = bs
            args.learning_rate = lr
            args.idx = f'{args.name}' + prefix + str(idx) #
            trainer = Trainer(args)
            best, _, save_path = trainer.train()
            row = {'best':best, 'lr': lr, 'bs': bs, 'save_path': save_path}
            if args.rank == 0 :
                log[len(log)]=row
                save_pd(args.log_file,log)
            del trainer
            idx +=1