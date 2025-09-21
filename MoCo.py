import torch 
from torch import nn 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from dataset import FER, ImbalancedDatasetSampler, DistributedSamplerWrapper, get_fer_transforms
from models import ImbalancedModel, compute_class_spherical_means, slerp, dim_dict, calc_class_mean
from torch import distributed as dist
from argparse import ArgumentParser
from Loss import Moco, KCL, compute_etf_loss
from opt import SAM
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from utils import get_acc, get_exp_id, get_ldmk, sync_scalar, measure_grad, concat_all_gather, sync_defaultdict
from collections import defaultdict
from datetime import timedelta
from aligners import get_aligner
from copy import deepcopy
import os 
import wandb


def include(loss, losses):
    to_check = loss.split('_')
    for loss_name in losses : 
        for loss in to_check : 
            if loss_name == loss : 
                return True 
    return False 

def get_model(args):
    model_params = {
        'num_classes': args.num_classes, 
        'model_type': args.model_type, 
        'feature_branch': args.feature_branch,
        'feature_module': args.feature_module, 
        'regular_simplex': False, 
        'cos': True, 
        'learnable_input_dist': False, 
        'input_layer': False, 
        'freeze_backbone': False, 
        'remain_backbone': False, 
        'decomposition': False,
        'img_size': args.img_size
    }
    model = ImbalancedModel(**model_params)
    aligner = get_aligner('checkpoint/adaface_vit_base_kprpe_webface12m').cuda() if 'kprpe' in args.model_type else None
    return model.cuda() if args.world_size ==1 else DDP(model.cuda(), device_ids=[args.local_rank], find_unused_parameters=True), \
         aligner, model_params

def get_loaders(args):
    train_transform, valid_transform, train_transform_wo_aug = get_fer_transforms(train=True,model_type=args.model_type), get_fer_transforms(train=False,model_type=args.model_type), get_fer_transforms(train=False,model_type=args.model_type)
    train_dataset, valid_dataset, train_dataset_wo_aug = FER(args=args, train=True, transform=train_transform, idx=False), FER(args=args, train=False, transform=valid_transform, idx=False), FER(args=args, train=False, transform=train_transform_wo_aug, idx=False, balanced=False)
    if args.world_size > 1 :
        if args.use_sampler :
            train_sampler = DistributedSamplerWrapper(ImbalancedDatasetSampler(train_dataset, labels=train_dataset.labels), shuffle=True)
        else:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
        valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
        train_sampler_wo_aug = DistributedSampler(train_dataset_wo_aug, shuffle=False)
    else:
        if args.use_sampler :
            train_sampler = ImbalancedDatasetSampler(train_dataset, labels=train_dataset.labels)
        else:
            train_sampler = None
        valid_sampler = None
        train_sampler_wo_aug = None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=args.num_workers, pin_memory=True)
    train_loader_wo_aug = DataLoader(train_dataset_wo_aug, batch_size=args.batch_size, sampler=train_sampler_wo_aug, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return train_loader, valid_loader, train_loader_wo_aug

def get_optimizer(args, model):
    opt = SAM(model.parameters(),base_optimizer=torch.optim.AdamW,lr=args.learning_rate,weight_decay=args.weight_decay)
    scheduler = ExponentialLR(opt,gamma=0.98)
    return opt, scheduler

def get_moco(args, model, loader, aligner=None):
    if args.mean_weight is not None and not os.path.exists(os.path.join(args.mean_weight, 'init_queue.pth')):
        m = model if args.world_size == 1 else model.module 
        mean = compute_class_spherical_means(loader, m, aligner,device=torch.device('cuda'), num_classes=args.num_classes, subset=args.moco_k)
    elif args.mean_weight is not None and os.path.exists(os.path.join(args.mean_weight, 'init_queue.pth')):
        mean = torch.load(os.path.join(args.mean_weight, 'init_queue.pth'), weights_only=False, map_location=torch.device('cuda'))
    init_queue = None if args.mean_weight is None else mean
    moco = Moco(args, deepcopy(model if args.world_size==1 else model.module), num_classes=args.num_classes,
     dim=dim_dict[args.model_type][-1 if args.feature_branch else 0], init_queue=init_queue)
    kcl = KCL(temperature=args.temperature, include_positives_in_denominator=args.include_positives_in_denominator,
     exclude_same_class_from_negatives=args.exclude_same_class_from_negatives, use_batch_negatives=args.use_batch_negatives)
    return moco, kcl

def get_args():
    args = ArgumentParser()

    # training hyperparameters 
    args.add_argument('--learning_rate', type=float, required=True)
    args.add_argument('--batch_size', type=int, required=True)
    args.add_argument('--n_epochs', type=int, required=True)
    args.add_argument('--weight_decay', type=float, required=True)

    # dataset info
    args.add_argument('--dataset_name', type=str, choices=['RAF-DB', 'AffectNet'], required=True)
    args.add_argument('--dataset_path', type=str, required=True)
    args.add_argument('--num_classes', type=int, default=7)
    args.add_argument('--use_sampler', default=False)
    args.add_argument('--img_size', type=int, choices=[112,224], default=112)

    # ckpts
    args.add_argument('--mean_weight', type=str, default=None)
    
    # model info 
    args.add_argument('--model_type', type=str, choices=['ir50', 'kprpe12m', 'kprpe4m', 'fmae_small', 'Pyramid_ir50'], required=True)
    args.add_argument('--feature_branch', default=False)
    args.add_argument('--feature_module', default=False, help='deepcomplex_depth, residual_depth')

    # distributed setting 
    args.add_argument('--world_size', type=int, default=1)
    args.add_argument('--local_rank', type=int, default=None)
    args.add_argument('--rank', type=int, default=None)
    args.add_argument('--num_workers', type=int, default=0)
    args.add_argument('--use_tf', default=False)

    # CL args
    args.add_argument('--loss', type=str, default='CE', help='first argument CE KBCL KCL second option ETF ex KCL_ETF')
    args.add_argument('--kcl_k', type=int, default=5)
    args.add_argument('--include_positives_in_denominator', default=False)
    args.add_argument('--exclude_same_class_from_negatives', default=False)
    args.add_argument('--use_batch_negatives', default=False)
    args.add_argument('--beta', type=float, default=0.3)
    args.add_argument('--etf_weight', type=float, default=0.0)
    args.add_argument('--temperature', type=float, default=0.1)
    args.add_argument('--utilze_class_centers', default=False)
    args.add_argument('--moco_k', type=int, default=2)

    # logging args 
    args.add_argument('--measure_grad', default=False)
    args.add_argument('--debug', default=False)

    args = args.parse_args()
    vars(args)['server'] = os.getenv('SERVER','0')
    if args.world_size > 1 :
        dist.init_process_group('nccl',world_size=args.world_size,
                           timeout=timedelta(minutes=60))
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.rank = int(os.environ['RANK'])
        args.batch_size = args.batch_size // args.world_size
        torch.cuda.set_device(args.local_rank)
    else:
        torch.cuda.set_device(0)

    if args.use_tf:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    return args

class Trainer:
    def __init__(self, args):
        self.args = args 
        self.model, self.aligner, self.model_params = get_model(args)
        self.train_loader, self.valid_loader, self.train_loader_wo_aug = get_loaders(args)
        if include(self.args.loss, ['KCL', 'KBCL']):
            self.moco, self.kcl = get_moco(args, self.model, self.train_loader_wo_aug,aligner=self.aligner)
            self.init_weight()

        self.opt, self.scheduler = get_optimizer(args, self.model)
        self.init_logs()

    
    def init_logs(self):
        if self.args.world_size == 1 or self.args.rank == 0 : 
            id = get_exp_id(self.args)
            wandb.login()
            wandb.init(
                project=f'CL-{self.args.dataset_name}', 
                id=id,
                name=id, 
                config={**vars(self.args), **self.model_params}
            )
            self.id = id
            if not os.path.exists(f'checkpoint/{self.id}'):
                os.makedirs(f'checkpoint/{self.id}')
        if self.args.world_size > 1 :
            dist.barrier()
            obj_list = [self.id if self.args.rank == 0 else None]
            dist.broadcast_object_list(obj_list, src=0)
            self.id = obj_list[0]
        self.log = defaultdict(list)
        self.save_dir = f'checkpoint/{self.id}'
        self.best_loss = float('inf')
        self.best_acc = -float('inf')
    
    @torch.no_grad()
    def init_weight(self):
        mean = compute_class_spherical_means(self.train_loader_wo_aug, self.model if self.args.world_size==1 else \
             self.model.module, device=torch.device('cuda'), num_classes=self.args.num_classes, aligner=self.aligner)
        dist.all_reduce(mean, op=dist.ReduceOp.AVG) if self.args.world_size > 1 else None
        mean = torch.nn.functional.normalize(mean, dim=1)
        if self.args.world_size == 1:
            self.model.weight.data = mean.T 
        else:
            self.model.module.weight.data = mean.T
        self.mean = mean 

    def process_loss(self, loss):
        total_loss = 0
        for key, value in loss.items():
            total_loss += value
        return total_loss

    def get_cl_loss(self, q, label,c , ldmk=None, k=None, k_label=None):
        # calc temporal class centers 
        if self.args.utilze_class_centers:
            q_for_mu = concat_all_gather(q) if self.args.world_size > 1 else q
            label_for_mu = concat_all_gather(label) if self.args.world_size > 1 else label
            temp_mean, temp_mask = calc_class_mean(q_for_mu, label_for_mu, self.args.num_classes)
            temp_mean = slerp(self.mean[temp_mask==True].detach().clone(), temp_mean[temp_mask==True], 0.001)
            now_mean = self.mean.clone().detach()
            now_mean[temp_mask==True] = temp_mean 
            self.mean[temp_mask==True] = now_mean[temp_mask==True]
        label = torch.cat([label, torch.arange(self.args.num_classes, device=torch.device('cuda')).repeat(2 if self.args.utilze_class_centers else 1)], dim=0)
        q = torch.cat([q,now_mean,c],dim=0) if self.args.utilze_class_centers else torch.cat([q,c],dim=0)
        k, k_label = self.moco.get_k(label, self.args.kcl_k) if k is None else (k,k_label)
        cl_loss = self.kcl(features=q, labels=label, pos_feats=k, pos_labels=k_label)
        return cl_loss * self.args.beta, k, k_label

    def run_train_forward(self, img, label, ldmk=None, k=None, k_label=None, loss_for_log=None):
        logit, q, c = self.model(img, keypoint=ldmk, features=True)
        temp_loss = defaultdict(float)
        bs= q.shape[0]
        ce_loss = torch.nn.functional.cross_entropy(logit, label)
        temp_loss['CE']=ce_loss
        if loss_for_log is not None:
            loss_for_log['CE']+=ce_loss.detach().item()*bs 
        if self.args.loss =='CE' :
            return logit, self.process_loss(temp_loss), torch.tensor(0,device=torch.device('cuda')), k, k_label
        cl_loss, k, k_label = self.get_cl_loss(q, label, c, ldmk, k, k_label)
        temp_loss['CL']=cl_loss
        if loss_for_log is not None:
            loss_for_log['CL']+=cl_loss.detach().item()*bs
        if include(self.args.loss, ['ETF']) :
            etf_loss = compute_etf_loss(self.model.get_kernel().T if self.args.world_size==1 else self.model.module.get_kernel().T, self.args.etf_weight)
            temp_loss['ETF']=etf_loss
            if loss_for_log is not None:
                loss_for_log['ETF']+=etf_loss.detach().item()*bs
        return logit, self.process_loss(temp_loss), k, k_label
    
    def run_valid_forward(self, img, label, ldmk=None):
        logit = self.model(img, keypoint=ldmk, features=False)
        loss = torch.nn.functional.cross_entropy(logit, label)
        return loss, logit
            
    def run_train_epoch(self):
        self.model.train()

        total_acc = 0
        temp_grad = defaultdict(list)
        loss_for_log = defaultdict(float)
        for img,label in tqdm(self.train_loader, disable=self.args.world_size > 1 and self.args.rank != 0, 
         desc=f"training epoch {self.epoch} latest_acc: {(self.log['valid_acc'][-1] if len(self.log['valid_acc']) > 0 else 0):.4f} best_acc: {self.best_acc:.4f}"):
            img, label = img.cuda(), label.cuda()
            ldmk = get_ldmk(img, self.aligner) if self.aligner is not None else None 
            logit, loss, k, k_label = self.run_train_forward(img, label, ldmk, k=None, k_label=None, loss_for_log=loss_for_log)
            loss.backward()
            if self.args.measure_grad : 
                temp_grad = measure_grad(self.model, 0, 0, temp_grad, layer_names=['backbone'])
            self.opt.first_step(zero_grad=True)
            with torch.no_grad():
                total_acc += get_acc(logit,label)*label.shape[0]
            _, loss, _ , _ = self.run_train_forward(img,label,ldmk, k=k, k_label=k_label, loss_for_log=None)
            loss.backward()
            self.opt.second_step(zero_grad=True)
            if include(self.args.loss, ['KCL', 'KBCL']):
                self.moco.momentum_update(self.model if self.args.world_size==1 else self.model.module)
                self.moco.enqueue(img, label, ldmk)
            
        if self.args.world_size > 1 : 
            total_acc = sync_scalar(total_acc, torch.device('cuda'))
            loss_for_log = sync_defaultdict(loss_for_log, N=len(self.train_loader.dataset), normalize=False)
        return total_acc / len(self.train_loader.dataset), loss_for_log


    @torch.no_grad()
    def run_valid_epoch(self):
        self.model.eval()
        total_loss = 0
        total_acc = 0
        for img, label in tqdm(self.valid_loader, disable=self.args.world_size > 1 and self.args.rank != 0,
         desc=f"validating epoch {self.epoch} latest_acc: {(self.log['valid_acc'][-1] if len(self.log['valid_acc']) > 0 else 0):.4f} best_acc: {self.best_acc:.4f}"):
            img, label = img.cuda(), label.cuda()
            ldmk = get_ldmk(img, self.aligner) if self.aligner is not None else None 
            loss, logit = self.run_valid_forward(img, label, ldmk)
            total_loss += loss.detach().item()*label.shape[0]
            total_acc += get_acc(logit, label)*label.shape[0]
        if self.args.world_size > 1 :
            total_loss = sync_scalar(total_loss, torch.device('cuda'))
            total_acc = sync_scalar(total_acc, torch.device('cuda'))
        return total_acc / len(self.valid_loader.dataset), total_loss / len(self.valid_loader.dataset)

    def save(self, valid_acc):
        if valid_acc > self.best_acc : 
            self.best_acc = valid_acc
        
        if self.args.world_size == 1 or self.args.rank == 0 :
            m = self.model if self.args.world_size == 1 else self.model.module
            ckpt = {
                'model_state_dict' : m.state_dict(),
                'optimizer_state_dict' : self.opt.state_dict(),
                'scheduler_state_dict' : self.scheduler.state_dict() if self.scheduler is not None else None,
                'best_acc' : self.best_acc,
                'epoch' : self.epoch,
                'log' : self.log,
                'args' : self.args,
                'id' : self.id,
                'mean' : self.mean if include(self.args.loss, ['KCL', 'KBCL']) else None, 
                'model_params': self.model_params,
                'args' : self.args
            }
            torch.save(ckpt, f'{self.save_dir}/latest.pth')
            if valid_acc == self.best_acc : 
                torch.save(ckpt, f'{self.save_dir}/best.pth')

    def run_epoch(self):
        train_acc, train_loss_for_log = self.run_train_epoch()
        valid_acc, valid_loss = self.run_valid_epoch()
        self.log['train_acc'].append(train_acc)
        for key, value in train_loss_for_log.items():
            self.log[key].append(value)
        self.log['valid_acc'].append(valid_acc)
        self.log['valid_loss'].append(valid_loss)
        if self.args.world_size == 1 or self.args.rank == 0 :
                wandb.log({
                'train_acc': train_acc,
                'valid_acc': valid_acc,
                'valid_loss': valid_loss,
                'best_acc': self.best_acc,
                'epoch': self.epoch,
                **dict(train_loss_for_log)
                })
        self.save(valid_acc)


    def train(self):
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