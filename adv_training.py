import torch 
import torch.distributed as dist
from dataset import random_masking, point_block_mask, get_fer_transforms, FER, DistributedSamplerWrapper, ImbalancedDatasetSampler
from aligners import get_aligner
from argparse import ArgumentParser
from torch.distributed import init_process_group, barrier
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm 
from utils import sync_scalar, get_ldmk, get_exp_id, get_acc
from Loss.Adv import compute_adv_loss
import wandb 
import os 
from opt import SAM
from datetime import timedelta
from models import ImbalancedModel

def get_optimizer(model, args):
    opt = SAM(model.parameters(),base_optimizer=torch.optim.AdamW,lr=args.learning_rate,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt,gamma=0.98) if args.dataset_name == 'RAF-DB' else torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.n_epochs,eta_min=args.learning_rate/100)
    return opt, scheduler

def get_loaders(args):
    train_transform = get_fer_transforms(train=True)
    valid_transform = get_fer_transforms(train=False)
    train_dataset = FER(args=args, train=True, transform=train_transform, idx=True)
    valid_dataset = FER(args=args, train=False, transform=valid_transform, idx=True)
    if args.world_size > 1 : 
        train_sampler = DistributedSampler(train_dataset, shuffle=True) if not args.use_sampler \
            else DistributedSamplerWrapper(ImbalancedDatasetSampler(train_dataset, labels=train_dataset.labels), shuffle=True)
        valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
    else:
        train_sampler  = ImbalancedDatasetSampler(train_dataset, labels=train_dataset.labels) if args.use_sampler else None
        valid_sampler = None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=args.num_workers, pin_memory=True)
    return train_loader, valid_loader

def get_model(args):
    model =  ImbalancedModel(num_classes=7, model_type=args.model_type, feature_branch=False, 
    feature_module=False, regular_simplex=False, cos=True, decomposition=True, learnable_input_dist=True, input_layer=False,
    freeze_backbone=False, remain_backbone=False)
    return model.cuda() if args.world_size ==1 else DDP(model.cuda(), device_ids=[args.local_rank], find_unused_parameters=True)

def get_args():
    args = ArgumentParser()
    # model args 
    args.add_argument('--model_type', choices=['ir50', 'kp_rpe'])
    
    # dataset_args 
    args.add_argument('--dataset_name', choices=['RAF-DB', 'AffectNet'])
    args.add_argument('--dataset_path', type=str)

    # research args 
    args.add_argument('--id_strategy', choices=['FR-clustering', 'masking'])
    args.add_argument('--n_blocks', type=int, default=10)
    args.add_argument('--detach_lowlevel', default=False)
    args.add_argument('--beta', type=float )
    args.add_argument('--partial_update', default=False)

    # training_args 
    args.add_argument('--learning_rate', type=float)
    args.add_argument('--batch_size', type=int)
    args.add_argument('--n_epochs', type=int)
    args.add_argument('--use_sampler', default=False)

    # distributed training
    args.add_argument('--world_size', type=int, default=1)
    args.add_argument('--local_rank', type=int, default=None)
    args.add_argument('--rank', type=int, default=None)
    args.add_argument('--num_workers', type=int, default=0)
    args.add_argument('--use_tf', default=False)
    
    # logging args 
    args.add_argument('--server', type=str)
    
    args = args.parse_args()
    vars(args)['server'] = os.getenv('SERVER','0')
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
    def __init__(self, args):
        self.args = args 
        self.aligner = get_aligner('checkpoint/adaface_vit_base_kprpe_webface12m').cuda() if args.model_type == 'kp_rpe' or args.id_strategy == 'masking' else None
        self.model = get_model(args)
        self.train_loader, self.valid_loader = get_loaders(args)
        self.init_logs()

        self.opt, self.scheduler = get_optimizer(self.model, self.args)

        
    def init_logs(self,):
        if self.args.world_size == 1 or self.args.rank == 0:
            id = get_exp_id(self.args)
            wandb.init(
                project=f'Adv-Training-{self.args.dataset_name}',
                id=id,
                name=id,
                config=self.args
            )
        if getattr(self.args, 'world_size', 1) > 1 and dist.is_available() and dist.is_initialized():
            obj_list = [id] if getattr(self.args, 'rank', 0) == 0 else [None]
            dist.broadcast_object_list(obj_list, src=0)
            id = obj_list[0]
        self.id = id 
        self.log = {}
        self.best_acc = 0
        if not os.path.exists(f'checkpoint/{self.id}') and (self.args.world_size == 1 or self.args.rank == 0):
            os.makedirs(f'checkpoint/{self.id}')
        
    def transform(self, img, ldmk):
        if self.args.id_strategy == 'masking':
            bs,_,h,_ = img.shape
            U_mask = random_masking(bs=bs, img_size=h, block_size=7, n_blocks=self.args.n_blocks,device=torch.device('cuda')) 

            L_mask = point_block_mask(bs=bs,ldmks=ldmk,img_size=h,n_blocks=self.args.n_blocks,block_size=7)
            anchor_img = img * U_mask.unsqueeze(1)
            neg_img = img * L_mask.unsqueeze(1)
            return anchor_img, neg_img 
        elif self.args.id_strategy == 'FR-clustering':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def backp(self, ce_loss, fr_loss):
        to_backp = self.model.module if self.args.world_size > 1 else self.model
        ce_loss.backward(retain_graph=True)
        torch.autograd.backward(tensors=[fr_loss*self.args.beta],
        inputs=(list(to_backp.backbone.body3.parameters())+list(to_backp.decomposition.parameters())) if self.args.partial_update \
             else list(to_backp.parameters()))


    def run_train_forward(self, img, label):
        img = img.cuda() # masking 등이 없는 img 임. 
        label = label.cuda()
        bs = label.shape[0]
        ldmk = get_ldmk(img, self.aligner) if self.aligner is not None else None
        anchor_img, neg_img = self.transform(img, ldmk) 
        entire = torch.concat([anchor_img, neg_img], dim=0)
        entire_ldmk = ldmk.repeat(2,1,1) if (ldmk is not None and self.args.model_type == 'kp_rpe') else None
        logits, fer_features, fr_features = self.model(entire, keypoint=entire_ldmk, features=True)
        anchor_fr_features, neg_fr_features = torch.split(fr_features, [bs,bs],dim=0)
        logits, _ = torch.split(logits, [bs,bs],dim=0)
        ce_loss = torch.nn.functional.cross_entropy(logits,label)
        fr_loss = compute_adv_loss(anchor_fr_features, neg_fr_features)
        return ce_loss, fr_loss, logits 

    def run_valid_forward(self, img, label):
        img = img.cuda()
        label = label.cuda()
        ldmk = get_ldmk(img, self.aligner) if (self.aligner is not None and self.args.model_type == 'kp_rpe') else None
        logits = self.model(img, keypoint=ldmk) 
        loss = torch.nn.functional.cross_entropy(logits,label)
        return loss, logits 
    
    def run_train_epoch(self):
        self.model.train()
        total_ce_loss = 0 
        total_fr_loss = 0 
        total_acc = 0 
        for img, label, idx in tqdm(self.train_loader, disable=self.args.world_size > 1 and self.args.rank != 0):
            self.model.zero_grad()
            ce_loss, fr_loss, logits = self.run_train_forward(img,label)
            self.backp(ce_loss, fr_loss)
            self.opt.first_step(zero_grad=True)
            ce_loss, fr_loss, logits = self.run_train_forward(img,label)
            self.backp(ce_loss, fr_loss)
            self.opt.second_step(zero_grad=True)
            bs = label.shape[0]
            total_ce_loss += ce_loss.detach().item() * bs
            total_fr_loss += fr_loss.detach().item() * bs
            total_acc += get_acc(logits, label.cuda()) * bs
        
        if self.args.world_size > 1 :
            total_ce_loss = sync_scalar(torch.tensor(total_ce_loss, device=torch.device('cuda')))
            total_fr_loss = sync_scalar(torch.tensor(total_fr_loss, device=torch.device('cuda')))
            total_acc = sync_scalar(torch.tensor(total_acc, device=torch.device('cuda')))
        return total_ce_loss / len(self.train_loader.dataset), total_fr_loss / len(self.train_loader.dataset), total_acc / len(self.train_loader.dataset)
    
    def run_valid_epoch(self):
        self.model.eval()
        total_loss = 0 
        total_acc = 0 
        for img, label, idx in tqdm(self.valid_loader, disable=self.args.world_size > 1 and self.args.rank != 0):
            loss, logits = self.run_valid_forward(img,label)
            bs = label.shape[0]
            total_loss += loss.detach().item() * bs
            total_acc += get_acc(logits, label.cuda()) * bs
        if self.args.world_size > 1 :
            total_loss = sync_scalar(torch.tensor(total_loss, device=torch.device('cuda')))
            total_acc = sync_scalar(torch.tensor(total_acc, device=torch.device('cuda')))
        return total_loss / len(self.valid_loader.dataset), total_acc / len(self.valid_loader.dataset)


    def save(self,valid_acc):
        is_best = False 
        if valid_acc > self.best_acc : 
            self.best_acc = valid_acc
            is_best = True 
        to_save = self.model if self.args.world_size == 1 else self.model.module
        ckpt = {
            'model_state_dict' : to_save.state_dict(),
            'optimizer_state_dict' : self.opt.state_dict(),
            'scheduler_state_dict' : self.scheduler.state_dict() if self.scheduler is not None else None,
            'best_acc' : self.best_acc,
            'epoch' : self.epoch,
            'log' : self.log,
            'args' : self.args,
            'id' : self.id,
        }
        if self.args.world_size == 1 or self.args.rank == 0:
            torch.save(ckpt, f'checkpoint/{self.id}/latest.pth')
            if is_best:
                torch.save(ckpt, f'checkpoint/{self.id}/best.pth')

    def run_epoch(self):
        train_ce_loss, train_fr_loss, train_acc = self.run_train_epoch()
        valid_loss, valid_acc = self.run_valid_epoch()
        self.scheduler.step()

        if self.args.world_size == 1 or self.args.rank == 0:
            wandb.log({
                'train_ce_loss': train_ce_loss,
                'train_fr_loss': train_fr_loss,
                'valid_loss': valid_loss,
                'train_acc': train_acc,
                'valid_acc': valid_acc,
            })
        self.save(valid_acc)

    def train(self):
        for epoch in range(self.args.n_epochs):
            self.epoch = epoch 
            self.run_epoch()

if __name__ == '__main__':
    args = get_args()
    trainer = Trainer(args)
    trainer.train()