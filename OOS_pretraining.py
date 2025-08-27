from dataset import get_kfolds, get_loaders
from models import make_g_nets
from argparse import ArgumentParser
from torch.distributed import init_process_group, barrier
import os, torch, pickle
import aligners 
from opt import SAM, WarmupCosineAdamW
import wandb 
from tqdm import trange, tqdm 
import numpy as np 
from datetime import datetime 
import pandas as pd 
from Loss import cosine_constant_margin_loss
from functools import partial

def get_exp_id(args):
    now = datetime.now()
    exp_id = args.server+now.strftime('%m%d%H%M%S%f')[:12]  # mmddhhmmssmm
    return exp_id

def get_args():
    args = ArgumentParser()
    #dataset config 
    args.add_argument('--dataset_path')
    args.add_argument('--dataset_name', choices=['clothing1m', 'AffectNet'])
    args.add_argument('--n_folds',type=int)
    args.add_argument('--random_seed',type=int)
    args.add_argument('--weight_decay',type=float)

    #training config
    args.add_argument('--learning_rate',type=float)
    args.add_argument('--batch_size',type=int)
    args.add_argument('--n_epochs',type=int)

    #distributed training
    args.add_argument('--world_size',type=int, default=1)
    args.add_argument('--local_rank',type=int)
    args.add_argument('--rank',type=int)


    #logging
    args.add_argument('--wandb_token',type=str)
    args.add_argument('--server',type=str)

    #model config
    args.add_argument('--kprpe_ckpt_path')
    args.add_argument('--architecture')
    args.add_argument('--pretrained', default=False)
    args.add_argument('--num_classes',type=int)

    #loss hyperparameters
    args.add_argument('--cos_constant_margin',type=float, default=False)

    args = args.parse_args()

    if args.world_size>1:
        init_process_group('nccl',world_size=args.world_size)
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.rank = int(os.environ['RANK'])
        args.batch_size = args.batch_size // args.world_size
        torch.cuda.set_device(args.local_rank)
        torch.cuda.empty_cache()
    return args

def get_optimizer(args):
    if args.dataset_name in ['AffectNet', 'RAF-DB']:
        return  partial(SAM,base_optimizer=torch.optim.AdamW, lr=args.learning_rate, weight_decay=args.weight_decay, adaptive=True)       
    elif args.dataset_name in ['clothing1m']:
        return partial(WarmupCosineAdamW, lr_max=args.learning_rate, lr_min=args.learning_rate/1000, warmup=20_000, total=400_000,weight_decay=args.weight_decay, betas=(0.9,0.999))
    else:
        raise ValueError(f"Dataset {args.dataset_name} not supported")
    
def get_scheduler(args):
    if args.dataset_name in ['AffectNet', 'RAF-DB']:
        return partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=args.n_epochs, eta_min=args.learning_rate/100)
    elif args.dataset_name in ['clothing1m']:
        return None
    else:
        raise ValueError(f"Dataset {args.dataset_name} not supported")

class Trainer:
    def __init__(self,args):
        self.args = args 
        
        # init datasets 
        self.train_sets, self.valid_sets = get_kfolds(
            n_folds=args.n_folds,
            args=args,
            random_seed=args.random_seed,
        )
        self.train_loaders, self.valid_loaders, self.train_samplers, self.valid_samplers = get_loaders(args,self.train_sets,self.valid_sets,use_ddp=args.world_size>1)

        # init model 
        self.device = torch.device(args.local_rank if args.world_size > 1 else 'cuda')
        
        self.g_nets = make_g_nets(args, self.device, pretrained=args.pretrained)
        if args.dataset_name in ['AffectNet', 'RAF-DB']:
            self.aligner = aligners.get_aligner(args.kprpe_ckpt_path).to(self.device)
            self.aligner.eval()
        
        # init optimizer 
        self.optimizers, self.schedulers = [], []
        for g_net in self.g_nets:
            m = g_net.module if args.world_size > 1 else g_net
            self.optimizers.append(get_optimizer(args)(m.parameters()))
            scheduler = get_scheduler(args)
            if scheduler is not None : 
                self.schedulers.append(scheduler(self.optimizers[-1]))
            else : 
                self.schedulers.append(None)

        self.loss_fn = cosine_constant_margin_loss if args.cos_constant_margin else torch.nn.functional.cross_entropy

        # init logs 
        if args.world_size == 1 or args.rank == 0:
            self.id = get_exp_id(args)
            self._init_wandb()
        self.best = [-1e9]*args.n_folds

        if args.world_size==1 or args.rank==0 and not os.path.exists(f'checkpoint/{self.id}'):
            os.mkdir(f'checkpoint/{self.id}')
            self.save_path = os.path.join(f'checkpoint', f'{self.id}')

        self.train_log = np.zeros((1,self.args.n_folds*2))
        self.valid_log = np.zeros((1,self.args.n_folds*2))
        
    def _init_wandb(self):
        token_path = getattr(self.args, 'wandb_token', None)
        if not os.environ.get('WANDB_API_KEY') and token_path:
            try:
                with open(token_path, 'r') as f:
                    api_key = f.readline().strip()
                if api_key:
                    os.environ['WANDB_API_KEY'] = api_key
            except Exception:
                pass
        try:
            wandb.login()
        except Exception:
            pass
        # Resume wandb run if run_id exists (for resuming training)
        wandb.init(
            project=f'OOS-pretraining-{self.args.dataset_name}',
            name=str(self.id),
            config=self.args,
            id=self.id,
            resume="allow" 
        )
    
    @torch.no_grad()
    def get_acc(self,preds,labels):
        return (preds.argmax(dim=1) == labels).float().mean().detach().cpu().item()
    
    def sync(self,metric):
        torch.distributed.all_reduce(metric, op=torch.distributed.ReduceOp.SUM)
        return metric.detach().cpu().item()
    
    def run_train_epoch(self, epoch):
        accs = []
        losses = []
        for idx, (g_net, opt, scheduler, train_loader, train_sampler) in enumerate(zip(self.g_nets, self.optimizers, self.schedulers, self.train_loaders, self.train_samplers)):
            if self.args.world_size > 1 :
                train_sampler.set_epoch(epoch)
            g_net.train()
            total_loss = 0
            total_acc = 0

            for img, label,_ in tqdm(train_loader, desc=f"epoch {epoch} fold {idx} Training", disable=(self.args.world_size > 1 and self.args.rank != 0)):
                opt.zero_grad()
                img, label = img.to(self.device), label.to(self.device)
                if self.args.dataset_name in ['AffectNet', 'RAF-DB']:
                    with torch.no_grad():
                        _,_,keypoint,_,_,_ = self.aligner(img)
                    preds = g_net(img, keypoint)
                else:
                    preds = g_net(img)
                loss = self.loss_fn(preds, label, self.args.cos_constant_margin)
                loss.backward()
                if self.args.dataset_name in ['AffectNet', 'RAF-DB']:
                    opt.first_step(zero_grad=True)
                    preds = g_net(img,keypoint)
                    loss = self.loss_fn(preds, label, self.args.cos_constant_margin)
                    loss.backward()
                    opt.second_step(zero_grad=True)
                else : 
                    opt.step()

                bs = label.shape[0]
                total_loss += loss.detach().cpu().item() * bs
                total_acc += self.get_acc(preds, label) * bs
            if self.args.world_size > 1:
                total_loss = self.sync(torch.tensor(total_loss, device=self.device))
                total_acc = self.sync(torch.tensor(total_acc, device=self.device))
            train_loss = total_loss/len(train_loader.dataset)
            train_acc = total_acc/len(train_loader.dataset)
            accs.append(train_acc)
            losses.append(train_loss)

            if scheduler is not None : 
                scheduler.step()

        return accs, losses
    
    @torch.no_grad()
    def run_valid_epoch(self, epoch):
        accs, losses = [], []
        for idx, (g_net, valid_loader) in enumerate(zip(self.g_nets, self.valid_loaders)):
            g_net.eval()
            total_loss = 0
            total_acc = 0

            for img, label, _ in tqdm(valid_loader, desc=f"epoch {epoch} fold {idx} Validating", disable=(self.args.world_size > 1 and self.args.rank != 0)):
                img, label = img.to(self.device), label.to(self.device)
                if self.args.dataset_name in ['AffectNet', 'RAF-DB']:
                    with torch.no_grad():
                        _,_,keypoint,_,_,_ = self.aligner(img)
                    pred = g_net(img,keypoint)
                else:
                    pred = g_net(img)
                loss = self.loss_fn(pred, label, self.args.cos_constant_margin)
                bs = label.shape[0]
                total_loss += loss.detach().cpu().item() * bs
                total_acc += self.get_acc(pred, label) * bs

            if self.args.world_size > 1:
                total_loss = self.sync(torch.tensor(total_loss, device=self.device))
                total_acc = self.sync(torch.tensor(total_acc, device=self.device))
            accs.append(total_acc/len(valid_loader.dataset))
            losses.append(total_loss/len(valid_loader.dataset))
        return accs, losses
    
    def run_epoch(self, epoch):
        train_accs, train_losses = self.run_train_epoch(epoch)
        valid_accs, valid_losses = self.run_valid_epoch(epoch)
        train_acc, train_loss = np.mean(train_accs), np.mean(train_losses)
        valid_acc, valid_loss = np.mean(valid_accs), np.mean(valid_losses)
        if self.args.world_size ==1 or self.args.rank ==0:
            self.save_ckpt(valid_accs)
            logs = {}
            for idx, (t_acc, v_acc) in enumerate(zip(train_accs, valid_accs)):
                logs[f'fold_{idx}_train_acc'] = t_acc
                logs[f'fold_{idx}_valid_acc'] = v_acc
                logs[f'fold_{idx}_train_loss'] = train_losses[idx]
                logs[f'fold_{idx}_valid_loss'] = valid_losses[idx]
            wandb.log({
                'train_acc': train_acc,
                'train_loss': train_loss,
                'valid_acc': valid_acc,
                'valid_loss': valid_loss,
                **logs 
            })
            train_log = train_accs + train_losses 
            valid_log = valid_accs + valid_losses 
            train_log = np.vstack((self.train_log, np.array(train_log).reshape(1,-1)))
            valid_log = np.vstack((self.valid_log, np.array(valid_log).reshape(1,-1)))
            columns = [f'fold_{i}_{name}' for i, name in zip(list(range(self.args.n_folds))*2,['acc']*self.args.n_folds+['loss']*self.args.n_folds)]
            train_log = pd.DataFrame(train_log, columns=columns).to_csv(os.path.join(self.save_path,'train_log.csv'), index=False)
            valid_log = pd.DataFrame(valid_log, columns=columns).to_csv(os.path.join(self.save_path,'valid_log.csv'), index=False)

        if self.args.world_size > 1:
            barrier()    
        
        return valid_acc 
            
    def save_ckpt(self, accs):
        for idx, acc in enumerate(accs):
            if acc > self.best[idx]:
                self.best[idx] = acc
                to_save = self.g_nets[idx].module if self.args.world_size > 1 else self.g_nets[idx]
                torch.save(to_save.state_dict(), os.path.join(self.save_path, f'fold_{idx}.pth'))
        checkpoint = {
            'opt':[opt.state_dict() for opt in self.optimizers],
            'sched':[scheduler.state_dict() if scheduler is not None else None for scheduler in self.schedulers],
            'best':self.best,
            'train_log':self.train_log,
            'valid_log':self.valid_log,
            'args':self.args,
            'epoch':self.epoch,
        }
        torch.save(checkpoint, os.path.join(self.save_path, f'checkpoint.pth'))
        
    def make_progress_bar(self, avg_acc):
        result = f'temp acc : {avg_acc} | best acc : {np.mean(self.best):.2f}'
        for idx, acc in enumerate(self.best):
            result += f' | fold_{idx}: {acc:.2f}'
        return result
    
    def train(self):
        pbar = trange(self.args.n_epochs, desc=f'start training', disable=(self.args.world_size > 1 and self.args.rank != 0))
        for epoch in pbar:
            self.epoch = epoch 
            val_acc = self.run_epoch(epoch)  # Assume run_epoch returns the most recent average validation accuracy
            # Update best if needed
            ment = self.make_progress_bar(val_acc)
            pbar.set_description(f'Epoch {epoch}, {ment}')


if __name__ == '__main__':
    args = get_args()
    trainer = Trainer(args)
    trainer.train()