import torch
import torch.distributed as dist
import time

from dataset import get_noise_dataset, get_kfolds, get_loaders, DistributedSamplerWrapper
from models import get_kprpe_pretrained, load_kprpe_finetuned, get_noise_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import get_acc, sync, sync_tensor
from argparse import ArgumentParser, Namespace
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from dataset import ImbalancedDatasetSampler
from opt import SAM, WarmupCosineAdamW
import wandb,os,pickle,aligners
from models import make_g_nets, load_g_nets, CosClassifier
from Loss.OOS_LNAAL import get_confidence_db, get_instant_margin, apply_margin
from datetime import timedelta, datetime
from functools import partial

def get_exp_id(args):
    now = datetime.now()
    exp_id = args.server+now.strftime('%m%d%H%M%S%f')[:12]  # mmddhhmmssmm
    return exp_id

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

def get_args():

    args = ArgumentParser()
    # for logs
    args.add_argument('--server',required=True,help='server name')
    args.add_argument('--wandb_token')
    args.add_argument('--use_tf',default=False)

    # training hyperparameteres
    args.add_argument('--learning_rate',type=float)
    args.add_argument('--batch_size',type=int)
    args.add_argument('--kprpe_ckpt_path')
    args.add_argument('--ckpt')
    args.add_argument('--n_epochs', type=int)
    args.add_argument('--training_checkpoint',default=None)
    args.add_argument('--g_net_dropout', default=False)
    args.add_argument('--weight_decay',type=float, default=0.05)
    args.add_argument('--architecture')
    args.add_argument('--pretrained',default=False)

    # for dataset
    args.add_argument('--dataset_path')
    args.add_argument('--dataset_name')
    args.add_argument('--num_classes',type=int)

    args.add_argument('--instance_ada_loss',default=False)
    args.add_argument('--cos_constant_margin',type=float)
    args.add_argument('--confidence_constant',type=float)
    args.add_argument('--cos_scaling',type=float, default=1)
    args.add_argument('--as_bias',default=False)

    # distributed training
    args.add_argument('--world_size',default=1,type=int)
    args.add_argument('--local_rank')
    args.add_argument('--rank')
    args.add_argument('--instance_ada_dropout',default=False)

    # OOS-arguments
    args.add_argument('--n_folds',type=int)
    args.add_argument('--g_net_ckpt')
    args.add_argument('--random_seed',type=int)
    args.add_argument('--oos_tensor',default=False)



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
        
    if args.use_tf : 
        torch.backends.cuda.matmul.allow_tf32 = True 
        torch.backends.cudnn.allow_tf32 = True 
        torch.backends.cudnn.benchmark = True 
        
    return args

class Trainer :
    def __init__(self,args):
        self.args = args if args.training_checkpoint is None else torch.load(args.training_checkpoint)['args']
        # init datasets for g_nets
        self.train_sets, self.valid_sets = get_kfolds(
            n_folds=args.n_folds,
            args=args,
            random_seed=args.random_seed
        )
        self.train_loaders, self.valid_loaders, self.train_samplers, self.valid_samplers = get_loaders(args,self.train_sets,self.valid_sets)
        # init datasets for target network 
        self.train_set = get_noise_dataset(args=args, train=True)
        train_sampler = ImbalancedDatasetSampler(self.train_set,labels=self.train_set.labels)
        if args.world_size > 1 :
            train_sampler = DistributedSamplerWrapper(train_sampler,shuffle=True)

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=16,
            sampler=train_sampler
        )
        self.valid_set = get_noise_dataset(args=args, train=False)
        valid_sampler = DistributedSampler(self.valid_set,shuffle=False) if args.world_size > 1 else None
        self.valid_loader = DataLoader(
            self.valid_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=16,
            sampler=valid_sampler
        )
        # init model 
        self.device = torch.device(f'cuda:{args.local_rank}' if args.world_size > 1 else 'cuda')
        self.model = get_noise_model(args, pretrained=args.pretrained)
        self.model = self.model.to(self.device)
        if args.world_size > 1 :
            self.model = DDP(self.model,device_ids=[args.local_rank],find_unused_parameters=True)
        if args.dataset_name in ['AffectNet', 'RAF-DB']:
            self.aligner = aligners.get_aligner(args.kprpe_ckpt_path)
            self.aligner = self.aligner.to(self.device)
            self.aligner.eval()
        else:
            self.aligner = None
        self.opt = get_optimizer(args)(self.model.parameters() if args.world_size == 1 else self.model.module.parameters())
        self.scheduler = get_scheduler(args)(self.opt)

        if args.instance_ada_loss or args.cos_margin_loss:
            self.g_nets = make_g_nets(args, self.device, freeze=True)
            self.g_nets = load_g_nets(self.g_nets, args.g_net_ckpt, self.device)
            if not args.g_net_dropout:
                for g_net in self.g_nets : 
                    g_net.eval()
            if os.path.exists(args.oos_tensor):
                if self.args.world_size == 1 or self.args.rank == 0:
                    print('Loading conf_db from', args.oos_tensor)
                    self.conf_db = torch.load(args.oos_tensor).contiguous()
                    shape = self.conf_db.shape
                else:
                    shape = None 
                if self.args.world_size > 1 :
                    dist.barrier()
            else:
                print('Computing conf_db from scratch')
                if self.args.world_size ==1 or self.args.rank ==0 :
                    print('Computing confidence db'); start = time.time()
                    self.conf_db = get_confidence_db(models=self.g_nets, aligner=self.aligner, loaders=self.valid_loaders, datasets=self.valid_sets, device=self.device).contiguous()
                    print('saving conf_db to', args.oos_tensor)
                    torch.save(self.conf_db, args.oos_tensor)
                    print(f'Time taken: {time.time() - start} seconds')
                    shape = self.conf_db.shape
                else:
                    shape = None
                if self.args.world_size > 1 :
                    dist.barrier()
            if self.args.world_size > 1 :
                shape = list(shape) if shape is not None else [0, 0] 
                shape_tensor = torch.tensor(shape, dtype=torch.long, device=self.device)
                dist.broadcast(shape_tensor, src=0)
                shape = tuple(shape_tensor.tolist())

                if self.args.rank != 0:
                    self.conf_db = torch.empty(shape, dtype=torch.float32, device=self.device)

                if self.args.world_size > 1 :
                    dist.barrier()
                    torch.distributed.broadcast(self.conf_db, src=0)
                    dist.barrier()
        print(f'dataset length : {len(self.train_set)} confdb_shape : {self.conf_db.shape}')
        self.best = -1e10
        if args.training_checkpoint is not None : 
            self.load_checkpoint()
        else:
            self.start_epoch = 0

        print(f"self.conf_db.shape: {self.conf_db.shape}")
        if args.world_size == 1 or args.rank == 0:
            self.id = get_exp_id(args)
            self._init_wandb()
            self.log = []
            print("self.conf_db.shape: ", self.conf_db.shape, "dataset size :", len(self.train_set))
            if not os.path.exists(f'checkpoint/{self.id}'):
                os.makedirs(f'checkpoint/{self.id}',exist_ok=True)

    def _init_wandb(self):
        if self.args.wandb_token:
            with open(self.args.wandb_token, 'r') as f:
                token = f.readline().strip()
            wandb.login(key=token)
            wandb.init(
                project=f'OOS-target_train-{self.args.dataset_name}',
                name=str(self.id),
                config=self.args,
                id=self.id,
                resume='allow'
            )
    

    def run_train_forward(self,img,label,ldmk,c):
        if ldmk is not None:
            pred = self.model(img, ldmk)
        else:
            pred = self.model(img)
        if self.args.instance_ada_loss:
            j = get_instant_margin(c,label)
            pred = apply_margin(pred,label,j,self.args.confidence_constant,self.args.as_bias)
            pred = self.args.cos_scaling * pred 
        loss = torch.nn.functional.cross_entropy(pred,label)
        return loss, pred 


    def run_train_epoch(self):
        train_acc = 0
        losses = 0
        for img, label, idx in tqdm(self.train_loader):
            self.model.zero_grad()
            img = img.to(self.device)
            label = label.to(self.device)
            c = self.conf_db[idx].to(self.device)
            bs = label.shape[0]
            if self.aligner is not None:
                with torch.no_grad():
                    _, _, ldmk, _, _, _ = self.aligner(img)
            else:
                ldmk = None
            loss,pred = self.run_train_forward(img,label,ldmk,c)
            loss.backward()
            with torch.no_grad():
                losses += (bs/len(self.train_set))*loss.item()
                train_acc += (bs / len(self.train_set)) * get_acc(pred, label)
            if self.aligner is not None:
                self.opt.first_step(zero_grad=True)
            else: 
                self.opt.step()
                continue
            loss,_ = self.run_train_forward(img,label,ldmk,c)
            loss.backward()
            self.opt.second_step(zero_grad=True)

        if self.args.world_size > 1:
            train_acc = sync(train_acc, self.device)
            losses =sync(losses, self.device)

        return train_acc, losses

    @torch.no_grad()
    def run_valid(self):
        acc = 0
        losses = 0
        for img, label, _ in tqdm(self.valid_loader):
            img = img.to(self.device)
            label = label.to(self.device)
            if self.aligner is not None:
                _, _, ldmk, _, _, _ = self.aligner(img)
                pred = self.model(img, ldmk)
            else:
                pred = self.model(img)
            loss = torch.nn.functional.cross_entropy(pred, label)
            bs = label.shape[0]
            acc += (bs / len(self.valid_set)) * (get_acc(pred, label))
            losses += (bs / len(self.valid_set)) * loss

        if self.args.world_size > 1:
            acc = sync(acc, self.device)
            losses = sync(losses, self.device)

        return acc,losses

    def save(self, train_acc, acc, valid_loss, train_loss, epoch):
        if self.args.world_size == 1 or self.args.rank == 0:
            wandb.log({'valid_acc': acc, 'train_acc': train_acc, 'train_loss': train_loss, 'valid_loss': valid_loss})
            to_save = self.model.module if self.args.world_size > 1 else self.model

            # Save latest checkpoint
            torch.save({
                'model_state_dict': to_save.state_dict(),
                'optimizer_state_dict': self.opt.state_dict() if self.scheduler is not None else None,
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
                'epoch': epoch,
                'best': self.best,
                'log': self.log,
                'args': self.args,
            }, f'checkpoint/{self.id}/model_latest.pt')

            # Save best checkpoint
            if self.best < acc:
                self.best = acc
                torch.save({
                    'model_state_dict': to_save.state_dict(),
                    'optimizer_state_dict': self.opt.state_dict() if self.scheduler is not None else None,
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
                    'epoch': epoch,
                    'best': self.best,
                    'log': self.log,
                    'args': self.args,
                }, f'checkpoint/{self.id}/model_best.pt')

    def run_epoch(self,epoch):
        if self.args.world_size > 1:
            self.train_loader.sampler.set_epoch(epoch)
            self.valid_loader.sampler.set_epoch(epoch)
        self.model.train()
        train_acc,train_loss = self.run_train_epoch()
        if self.scheduler is not None:
            self.scheduler.step()
        self.model.eval()
        valid_acc, valid_loss = self.run_valid()
        self.save(train_acc,valid_acc,valid_loss,train_loss,epoch)

    def load_checkpoint(self):
        latest_ckpt = f'checkpoint/{self.id}/model_latest.pt'
        if os.path.exists(latest_ckpt):
            checkpoint = torch.load(latest_ckpt, map_location=self.device)
            to_load = self.model.module if self.args.world_size > 1 else self.model
            to_load.load_state_dict(checkpoint['model_state_dict'])
            self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint.get('epoch', 0) + 1
            self.best = checkpoint.get('best', -1e10)
            self.log = checkpoint.get('log', [])
            print(f"Resumed from epoch {self.start_epoch}")
        else:
            self.start_epoch = 0

    def train(self):
        for epoch in range(self.start_epoch, self.args.n_epochs):
            self.run_epoch(epoch)
            if self.args.world_size == 1 or self.args.rank == 0 :
                with open(f'checkpoint/{self.id}/log.pkl','wb') as f:
                    pickle.dump(self.log,f)

if __name__ == '__main__':
    args = get_args()
    trainer = Trainer(args)
    trainer.train()