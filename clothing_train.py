import torch.optim.lr_scheduler as lr_scheduler
from dataset import Clothing1m
from models import get_resnet_classifier
from torch.utils.data import DataLoader, DistributedSampler
from argparse import ArgumentParser
from torch.distributed import init_process_group
import torch.distributed as dist
from opt import *
import wandb, os ,pickle
from dataset import ImbalancedDatasetSampler, DistributedSamplerWrapper
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import init_wandb_clothing1m, get_acc, sync
from tqdm import tqdm
from Loss import get_label_noise, including_margin

def get_args():
    args = ArgumentParser()
    # dataset
    args.add_argument('--dataset_path', type=str)
    args.add_argument('--dataset_name', type=str)

    # training hyper parameters
    args.add_argument('--batch_size', type=int)
    args.add_argument('--n_epochs', type=int)
    args.add_argument('--learning_rate', type=float)

    # for log
    args.add_argument('--wandb_token', type=str)
    args.add_argument('--server')
    args.add_argument('--name', type=str)

    # loss hyper parameters
    args.add_argument('--constant_margin', type=float, default=False)
    args.add_argument('--adaptive_margin', default=False, type=float)


    # for distributed training
    args.add_argument('--world_size',default=1,type=int)
    args.add_argument('--rank')
    args.add_argument('--local_rank')

    # checkpoint informations
    args.add_argument('--lq_ckpt')
    args.add_argument('--use_dropout',default=False)

    args =  args.parse_args()

    if args.world_size > 1 :
        args.rank = int(os.environ['RANK'])
        init_process_group('nccl', world_size=args.world_size, rank=args.rank)

        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.rank = int(os.environ['RANK'])
        args.batch_size =int(args.batch_size/args.world_size)
        torch.cuda.set_device(args.local_rank)
    return args

class Trainer:
    def __init__(self, args):

        # load dataset utils
        self.train_set = Clothing1m(args.dataset_path, train=True)
        self.valid_set = Clothing1m(args.dataset_path, train=False)
        self.train_sampler = ImbalancedDatasetSampler(self.train_set) if args.world_size ==1 else DistributedSamplerWrapper(ImbalancedDatasetSampler(self.train_set),shuffle=True)
        self.valid_sampler = DistributedSampler(self.valid_set, shuffle=False) if args.world_size > 1 else None

        self.train_loader = DataLoader(self.train_set, batch_size=args.batch_size,sampler=self.train_sampler)
        self.valid_loader = DataLoader(self.valid_set, batch_size=args.batch_size,sampler=self.valid_sampler)


        # load models
        self.device = torch.device('cuda') if args.world_size ==1 else args.local_rank
        self.model = get_resnet_classifier(num_classes=14).cuda()
        if args.world_size > 1 :
            self.model = DDP(self.model, device_ids=[args.local_rank],find_unused_parameters=True)
        print(args.lq_ckpt)
        if args.lq_ckpt :
            print('loading quality assesment model')
            self.lq_model = get_resnet_classifier(num_classes=14).cuda()
            self.lq_model.load_state_dict(torch.load(os.path.join(args.lq_ckpt,'best_acc_model.pth')))
            if args.use_dropout :
                self.lq_model.train()
            else:
                self.lq_model.eval()

        # init opts
        self.opt = SAM(self.model.parameters(),base_optimizer=torch.optim.AdamW,lr=args.learning_rate,weight_decay=args.learning_rate/5)
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.opt,args.n_epochs)


        # for logs
        self.args = args
        self.train_len = len(self.train_set)
        self.valid_len = len(self.valid_set)

        if args.world_size == 1 or args.rank ==0:
            init_wandb_clothing1m(args)
            self.log = {
                'train_acc':[],
                'train_loss':[],
                'valid_acc':[],
                'valid_loss':[],
            }

        self.best_acc = -torch.inf
        self.best_loss = torch.inf

        # Loss
        if self.args.constant_margin :
            self.loss_fn = self.insert_margin
        else:
            self.loss_fn = torch.nn.functional.cross_entropy

    def insert_margin(self, input, target):
        margin = torch.zeros_like(input).to(self.device)
        margin[torch.arange(target.shape[0]).int(),target.int()] = self.args.constant_margin
        return torch.nn.functional.cross_entropy(input=torch.nn.functional.softmax(input-margin,dim=1),target=target)

    def run_forward(self, imgs, labels, train=True):
        preds = self.model(imgs)
        if train and self.args.adaptive_margin :
            with torch.no_grad():
                lq_preds = self.lq_model(imgs)
                j = get_label_noise(label=labels,pred=lq_preds)
            preds = including_margin(cos=preds,j=j,m=self.args.adaptive_margin)

        loss = self.loss_fn(target=labels,input=preds)
        return loss, preds
    
    def run_train_epoch(self):
        acc = 0
        loss = 0
        for imgs, labels in tqdm(self.train_loader, desc=f'epoch: {self.epoch} training loop best_acc:{self.best_acc:.4f} best_loss:{self.best_loss:.4f}', disable=self.args.world_size>1 and self.args.rank!=0):
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            loss, preds = self.run_forward(imgs,labels)
            loss.backward()
            self.opt.first_step()
            with torch.no_grad():
                acc += (imgs.shape[0]/self.train_len)*get_acc(preds,labels)
                loss += loss.detach()*(imgs.shape[0]/self.train_len)
            self.opt.zero_grad()
            loss, preds = self.run_forward(imgs,labels)
            loss.backward()
            self.opt.second_step(zero_grad=True)
        if self.args.world_size > 1 :
            acc = sync(acc,device=self.device)
            loss = sync(loss,device=self.device)
        return acc, loss

    @torch.no_grad()
    def run_valid_epoch(self):
        acc = 0
        loss = 0
        for imgs, labels in tqdm(self.valid_loader, desc=f'epoch: {self.epoch} validation loop best_acc:{self.best_acc:.4f} best_loss:{self.best_loss:.4f}', disable=self.args.world_size>1 and self.args.rank!=0):
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            loss, preds = self.run_forward(imgs,labels,train=False)
            acc += (imgs.shape[0]/self.valid_len)*get_acc(preds,labels)
            loss += loss.detach()*(imgs.shape[0]/self.valid_len)
        if self.args.world_size > 1 :
            acc = sync(acc,device=self.device)
            loss = sync(loss,device=self.device)
        return acc, loss

    def run_epoch(self):
        self.model.train()
        if self.args.world_size > 1 :
            self.train_loader.sampler.set_epoch(self.epoch)
        train_acc, train_loss = self.run_train_epoch()
        self.model.eval()
        valid_acc, valid_loss = self.run_valid_epoch()

        if self.args.world_size ==1 or self.args.rank == 0 :
            self.save_logs(train_acc, train_loss, valid_acc, valid_loss)

        if self.args.world_size > 1 :
            dist.barrier()
        self.scheduler.step()
    def train(self):
        for epoch in range(self.args.n_epochs):
            self.epoch = epoch 
            self.run_epoch()

    def save_logs(self, train_acc, train_loss, valid_acc, valid_loss):
        # Save metrics to pkl
        self.log['train_acc'].append(train_acc)
        self.log['train_loss'].append(train_loss)
        self.log['valid_acc'].append(valid_acc)
        self.log['valid_loss'].append(valid_loss)
        
        save_path = os.path.join('checkpoint',f'{self.args.name}')
        if not os.path.exists(save_path) and (self.args.world_size == 1 or self.args.rank == 0):
            os.mkdir(save_path)
        
        with open(f'{save_path}/log.pkl','wb') as f :
            pickle.dump(self.log,f)

        # Log metrics to wandb
        wandb.log({
            'train/accuracy': train_acc,
            'train/loss': train_loss,
            'valid/accuracy': valid_acc, 
            'valid/loss': valid_loss,
        })

        to_save = self.model if self.args.world_size == 1 else self.model.module

        if valid_acc > self.best_acc :
            self.best_acc = valid_acc
            torch.save(to_save.state_dict(),f'{save_path}/best_acc_model.pth')
        if valid_loss < self.best_loss :
            self.best_loss = valid_loss
            torch.save(to_save.state_dict(),f'{save_path}/best_loss_model.pth')


if __name__ == '__main__':
    args = get_args()
    trainer = Trainer(args)
    trainer.train()