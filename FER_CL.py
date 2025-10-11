import torch 
from torch import nn 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from dataset import FER, ImbalancedDatasetSampler, DistributedSamplerWrapper, get_fer_transforms, ClassBatchSampler, get_multi_view_transforms
from models import ImbalancedModel, compute_class_spherical_means, slerp, dim_dict, calc_class_mean
from torch import distributed as dist
from argparse import ArgumentParser
from Loss import Moco, KCL, compute_etf_loss, EKCL, get_cl_loss
from opt import SAM
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from tqdm import tqdm
from utils import get_acc, get_exp_id, get_ldmk, sync_scalar, measure_grad, concat_all_gather, sync_defaultdict, get_mem, get_macro_acc, gather_tensor
from analysis import plot_angle_matrix
import wandb
from collections import defaultdict
from datetime import timedelta
from aligners import get_aligner
from copy import deepcopy
import os 
import shutil



def include(loss, losses):
    to_check = loss.split('_')
    for loss_name in losses : 
        for loss in to_check : 
            if loss_name == loss : 
                return True 
    return False 

def get_model(args):
    if args.ckpt_path is not None or args.resume_path is not None:
        path = args.ckpt_path if args.ckpt_path is not None else args.resume_path
        ckpt_path = os.path.join(path,'latest.pth')
        model_params = torch.load(ckpt_path, weights_only=False, map_location=torch.device('cpu'))['model_params']
        model = ImbalancedModel(**model_params)
        model.load_from_state_dict(ckpt_path, clear_weight=args.clear_classifier)
    else:
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
            'img_size': args.img_size,
            'use_bn': args.use_bn
        }
        model = ImbalancedModel(**model_params)
    aligner = get_aligner('checkpoint/adaface_vit_base_kprpe_webface12m').cuda() if 'kprpe' in args.model_type else None
    return model.cuda() if args.world_size ==1 else DDP(model.cuda(), device_ids=[args.local_rank], find_unused_parameters=True), \
         aligner, model_params

def get_loaders(args):
    train_transform, valid_transform, train_transform_wo_aug = get_multi_view_transforms(args, train=True,model_type=args.model_type), get_multi_view_transforms(args, train=False,model_type=args.model_type), get_multi_view_transforms(args, train=False,model_type=args.model_type)
    train_dataset, valid_dataset, train_dataset_wo_aug = FER(args=args, train=True, transform=train_transform, idx=False, imb_factor=args.imb_factor), FER(args=args, train=False, transform=valid_transform, idx=False), FER(args=args, train=False, transform=train_transform_wo_aug, idx=False, balanced=False,imb_factor=args.imb_factor)
    balanced_dataset = FER(args,transform=valid_transform, train=False, idx=False, balanced=True, imb_factor=args.imb_factor) if args.dataset_name == 'RAF-DB' else None
    
    if args.world_size > 1 : 
        if args.use_sampler :
            train_sampler = DistributedSamplerWrapper(ImbalancedDatasetSampler(train_dataset, labels=train_dataset.labels), shuffle=True)
        else:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
        valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
        train_sampler_wo_aug = DistributedSampler(train_dataset_wo_aug, shuffle=False)
        balanced_sampler = DistributedSampler(balanced_dataset, shuffle=False) if args.dataset_name == 'RAF-DB' else None
    else:
        if args.use_sampler :
            train_sampler = ImbalancedDatasetSampler(train_dataset, labels=train_dataset.labels)
        else:
            train_sampler = None
        valid_sampler = None
        train_sampler_wo_aug = None
        balanced_sampler = None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, shuffle=train_sampler is None)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=args.num_workers, pin_memory=True)
    train_loader_wo_aug = DataLoader(train_dataset_wo_aug, batch_size=args.batch_size, sampler=train_sampler_wo_aug, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    balanced_loader = DataLoader(balanced_dataset, batch_size=args.batch_size, sampler=balanced_sampler, shuffle=False, num_workers=args.num_workers, pin_memory=True) if args.dataset_name == 'RAF-DB' else None 
    return train_loader, valid_loader, train_loader_wo_aug, balanced_loader

def get_optimizer(args, model):
    opt = SAM(model.parameters(),base_optimizer=torch.optim.AdamW,lr=args.learning_rate,weight_decay=args.weight_decay)
    scheduler = ExponentialLR(opt,gamma=0.98) if args.scheduler=='exp' else CosineAnnealingLR(opt,T_max=args.n_epochs,eta_min=args.learning_rate/100)
    if args.resume_path is not None:
        ckpt = torch.load(os.path.join(args.resume_path, 'latest.pth'), map_location=torch.device('cpu'),weights_only=False)
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return opt, scheduler

def get_queue(args, model, loader, aligner=None):
    m = model if args.world_size == 1 else model.module 
    mean = compute_class_spherical_means(loader, m, aligner,device=torch.device('cuda'), num_classes=args.num_classes, subset=args.moco_k)
    return mean

def get_args():
    args = ArgumentParser()

    # distributed setting 
    args.add_argument('--world_size', type=int, default=1)
    args.add_argument('--num_workers', type=int, default=0)
    args.add_argument('--use_tf', default=False)
    args.add_argument('--local_rank', type=int, default=None)
    args.add_argument('--rank', type=int, default=None)

    # training hyperparameters 
    args.add_argument('--learning_rate', type=float, required=True)
    args.add_argument('--batch_size', type=int, required=True)
    args.add_argument('--n_epochs', type=int, required=True)
    args.add_argument('--weight_decay', type=float, required=True)
    args.add_argument('--optimizer', type=str, choices=['AdamW','SAM'], default='SAM')
    args.add_argument('--scheduler', choices=['exp', 'cosine'] , default='exp')

    
    # dataset info
    args.add_argument('--dataset_name', type=str, choices=['RAF-DB', 'AffectNet', 'CAER'], required=True)
    args.add_argument('--dataset_path', type=str, required=True)
    args.add_argument('--num_classes', type=int, default=7)
    args.add_argument('--use_sampler', default=False)
    args.add_argument('--img_size', type=int, choices=[112,224], default=112)
    args.add_argument('--use_view', default=False)
    args.add_argument('--imb_factor', type=float, default=1.0)


    # ckpts

    args.add_argument('--resume_path', type=str, default=None)
    
    # model info 
    args.add_argument('--model_type', type=str, choices=['ir50', 'kprpe12m', 'kprpe4m', 'fmae_small', 'Pyramid_ir50'], required=True)
    args.add_argument('--feature_branch', default=False)
    args.add_argument('--feature_module', default=False, help='deepcomplex_depth, residual_depth')
    args.add_argument('--ckpt_path', type=str, default=None )
    args.add_argument('--clear_classifier', default=False)
    args.add_argument('--init_classifier', default=False)
    args.add_argument('--use_bn', default=False)


    # CL args
    args.add_argument('--loss', type=str, default='CE', help='first argument CE KBCL KCL second option ETF ex KCL_ETF')
    args.add_argument('--kcl_k', type=int, default=5)
    args.add_argument('--include_positives_in_denominator', default=False)
    args.add_argument('--exclude_same_class_from_negatives', default=False)
    args.add_argument('--use_batch_negatives', default=False)
    args.add_argument('--except_sam', default=False)
    args.add_argument('--k_meeting_dist', default=False, type=float)

    args.add_argument('--beta', type=float, default=0.3)

    args.add_argument('--temperature', type=float, default=0.1)

    args.add_argument('--moco_k', type=int, default=2)
    args.add_argument('--k_meeting', type=str, default=None)

    args.add_argument('--k_grad', default=False)
    args.add_argument('--balanced_cl', default=False)
    args.add_argument('--utilize_class_centers', default=False)
    args.add_argument('--utilize_target_centers', default=False)
    
    args.add_argument('--etf_weight', type=float, default=0.0)
    args.add_argument('--etf_statistics', default=False)
    args.add_argument('--etf_std', type=float, default=0.7)

    # logging args 
    args.add_argument('--measure_grad', default=False)
    args.add_argument('--debug', default=False)


    args = args.parse_args()
    vars(args)['server'] = os.getenv('SERVER','0')
    vars(args)['dim'] = dim_dict[args.model_type][-1 if args.feature_branch else 0]
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
        self.train_loader, self.valid_loader, self.train_loader_wo_aug, self.balanced_loader = get_loaders(args)
        self.init_weight() if (not args.loss == 'CE' or args.resume_path is not None) else None 
        self.opt, self.scheduler = get_optimizer(args, self.model)
        self.init_logs()
        self.cl_loss = get_cl_loss(args, deepcopy(self.model if self.args.world_size==1 else self.model.module)
        , init_queue=get_queue(args, self.model, self.train_loader_wo_aug,aligner=self.aligner) if not include(self.args.loss, ['KCL', 'KBCL']) else None) 
    
    def init_logs(self):
        if self.args.resume_path is not None:
            ckpt = torch.load(os.path.join(self.args.resume_path, 'latest.pth'), map_location=torch.device('cpu'), weights_only=False)
        if self.args.world_size == 1 or self.args.rank == 0 : 
            id = get_exp_id(self.args)
            id = ckpt['id'] if self.args.resume_path is not None else id
            wandb.login()
            wandb.init(
                project=f'CL-{self.args.dataset_name}', 
                id=id,
                name=id, 
                resume='allow', 
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
        self.best_acc = -float('inf') if not self.args.resume_path is not None else ckpt['best_acc']
        self.best_macro_acc = -float('inf') if not self.args.resume_path is not None else (ckpt['best_macro_acc'] if 'best_macro_acc' in ckpt.keys() else -float('inf'))
        self.epoch = 0 if not self.args.resume_path is not None else ckpt['epoch'] + 1
    
    @torch.no_grad()
    def init_weight(self):
        mean = compute_class_spherical_means(self.train_loader_wo_aug, self.model if self.args.world_size==1 else \
             self.model.module, device=torch.device('cuda'), num_classes=self.args.num_classes, aligner=self.aligner)
        dist.all_reduce(mean, op=dist.ReduceOp.AVG) if self.args.world_size > 1 else None
        mean = torch.nn.functional.normalize(mean, dim=1)
        if self.args.init_classifier : 
            if self.args.world_size == 1:
                self.model.init_weight(mean.T)
            else:
                self.model.module.init_weight(mean.T)
        self.mean = mean 

    def process_loss(self, loss):
        total_loss = 0
        for key, value in loss.items():
            if value is not None:
                total_loss += value
        return total_loss

    def get_cl_loss(self, logit, q, label ,c , ldmk=None, k=None):

        if self.args.utilize_class_centers:
            q_for_mu = gather_tensor(q) if self.args.world_size > 1 else q
            label_for_mu = gather_tensor(label) if self.args.world_size > 1 else label
            temp_mean, temp_mask = calc_class_mean(q_for_mu, label_for_mu, self.args.num_classes)
            temp_mean = slerp(self.mean[temp_mask==True].detach().clone(), temp_mean[temp_mask==True], 0.001)
            now_mean = self.mean.clone().detach()
            now_mean[temp_mask==True] = temp_mean 
            self.mean[temp_mask==True] = now_mean[temp_mask==True].detach().clone()
            # returns aggregated class centers ( instance ) 
        else:
            now_mean = None

        weight = c if self.args.utilize_target_centers else None


        ce_loss, cl_loss, k = self.cl_loss(logits=logit, features=q, y=label, weight=weight, centers=now_mean,
        model=self.model if self.args.world_size==1 else self.model.module ,aligner=self.aligner, positive_pair=k, requires_grad=self.args.k_grad)

        return ce_loss, cl_loss*self.args.beta, k

    def run_train_forward(self, img, label, ldmk=None, k=None, loss_for_log=None, ce_only=False):
        logit, q, c = self.model(img, keypoint=ldmk, features=True)
        temp_loss = defaultdict(float)
        bs= q.shape[0]
        ce_loss, cl_loss, k = self.get_cl_loss(logit, q, label, c, ldmk, k=k) if not ce_only else (torch.nn.functional.cross_entropy(logit, label), None, None)
        temp_loss['CE']=ce_loss
        temp_loss['CL']=cl_loss
        if loss_for_log is not None:
            loss_for_log['CL']+= cl_loss.detach().item()*bs if cl_loss is not None else 0
            loss_for_log['CE']+=ce_loss.detach().item()*bs 
        if include(self.args.loss, ['ETF']) and not ce_only :
            etf_loss = compute_etf_loss(self.model.get_kernel().T if self.args.world_size==1 else self.model.module.get_kernel().T, self.args.etf_weight,
             statistics=self.args.etf_statistics, std_weight=self.args.etf_std)
            temp_loss['ETF']=etf_loss
            if loss_for_log is not None:
                loss_for_log['ETF']+=etf_loss.detach().item()*bs
        return logit[:label.shape[0]], self.process_loss(temp_loss), k
    
    @torch.inference_mode()
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
            self.model.zero_grad()
            if isinstance(img, list) : 
                img = torch.concat(img, dim=0)
            img, label = img.cuda(), label.cuda() # the images are list with number-of-views 
            ldmk = get_ldmk(img, self.aligner) if self.aligner is not None else None 
            logit, loss, k = self.run_train_forward(img, label, ldmk, k=None, loss_for_log=loss_for_log, ce_only=self.args.except_sam)
            loss.backward()
            if self.args.measure_grad : 
                temp_grad = measure_grad(self.model, 0, 0, temp_grad, layer_names=['backbone'])
            self.opt.first_step(zero_grad=True)
            with torch.no_grad():
                total_acc += get_acc(logit,label)*label.shape[0]
            _, loss, _ = self.run_train_forward(img,label,ldmk, k=k, loss_for_log=None)
            loss.backward()
            self.opt.second_step(zero_grad=True)

            if include(self.args.loss, ['KCL', 'KBCL']):
                self.cl_loss.momentum_update(self.model if self.args.world_size==1 else self.model.module)
                self.cl_loss.enqueue(img, label, ldmk)
            
        if self.args.world_size > 1 : 
            total_acc = sync_scalar(total_acc)
            loss_for_log = sync_defaultdict(loss_for_log, N=len(self.train_loader.dataset), normalize=False)
        else: 
            for key, value in loss_for_log.items():
                loss_for_log[key] = value / len(self.train_loader.dataset)
        return total_acc / len(self.train_loader.dataset), loss_for_log


    @torch.inference_mode()
    def run_valid_epoch(self):
        self.model.eval()
        total_loss = 0
        total_acc = 0
        total_macro_acc = torch.zeros((self.args.num_classes),device=torch.device('cuda')).float()
        balanced_loss = 0
        balanced_acc = 0
        for img, label in tqdm(self.valid_loader, disable=self.args.world_size > 1 and self.args.rank != 0,
         desc=f"validating epoch {self.epoch} latest_acc: {(self.log['valid_acc'][-1] if len(self.log['valid_acc']) > 0 else 0):.4f} best_acc: {self.best_acc:.4f}"):
            img, label = img.cuda(), label.cuda()
            ldmk = get_ldmk(img, self.aligner) if self.aligner is not None else None 
            loss, logit = self.run_valid_forward(img, label, ldmk)
            total_loss += loss.detach().item()*label.shape[0]
            total_acc += get_acc(logit, label)*label.shape[0]
            total_macro_acc += get_macro_acc(logit, label)

        if self.args.dataset_name == 'RAF-DB':
            for img, label in tqdm(self.balanced_loader, disable=self.args.world_size > 1 and self.args.rank != 0,
             desc=f"validating epoch {self.epoch} latest_acc: {(self.log['valid_acc'][-1] if len(self.log['valid_acc']) > 0 else 0):.4f} best_acc: {self.best_acc:.4f}"):
                img, label = img.cuda(), label.cuda()
                ldmk = get_ldmk(img, self.aligner) if self.aligner is not None else None 
                loss, logit = self.run_valid_forward(img, label, ldmk)
                balanced_loss += loss.detach().item()*label.shape[0]
                balanced_acc += get_acc(logit, label)*label.shape[0]

        if self.args.world_size > 1 :
            total_loss = sync_scalar(total_loss)
            total_acc = sync_scalar(total_acc)
            dist.all_reduce(total_macro_acc, op=dist.ReduceOp.SUM)
            balanced_loss = sync_scalar(balanced_loss) if self.balanced_loader is not None else 0
            balanced_acc = sync_scalar(balanced_acc) if self.balanced_loader is not None else 0

        N = len(self.valid_loader.dataset) 
        NB = len(self.balanced_loader.dataset) if self.balanced_loader is not None else 1
        total_macro_acc = (total_macro_acc / torch.tensor(self.valid_loader.dataset.get_img_num_per_cls(), device=torch.device('cuda'), dtype=torch.float32)).float().mean().detach().cpu().item()
        return total_acc / N, total_loss/N, balanced_acc / NB, balanced_loss / NB, total_macro_acc

    def save(self, valid_acc, valid_macro_acc):
        if valid_acc > self.best_acc : 
            self.best_acc = valid_acc
        if valid_macro_acc > self.best_macro_acc : 
            self.best_macro_acc = valid_macro_acc
        if self.args.world_size == 1 or self.args.rank == 0 :
            m = self.model if self.args.world_size == 1 else self.model.module
            ckpt = {
                'model_state_dict' : m.state_dict(),
                'optimizer_state_dict' : self.opt.state_dict(),
                'scheduler_state_dict' : self.scheduler.state_dict() if self.scheduler is not None else None,
                'best_acc' : self.best_acc,
                'best_macro_acc' : self.best_macro_acc,
                'epoch' : self.epoch,
                'log' : self.log,
                'args' : self.args,
                'id' : self.id,
                'mean' : self.mean if include(self.args.loss, ['KCL', 'KBCL']) else None, 
                'model_params': self.model_params,
                'args' : self.args,
                'moco' : self.cl_loss.moco.key_encoder.state_dict() if include(self.args.loss, ['KCL', 'KBCL']) else None,
                'moco_queue' : self.cl_loss.moco.queue if include(self.args.loss, ['KCL', 'KBCL']) else None
            }
            torch.save(ckpt, f'{self.save_dir}/latest.pth')
            if valid_acc == self.best_acc : 
                torch.save(ckpt, f'{self.save_dir}/best.pth')
            if valid_macro_acc == self.best_macro_acc : 
                torch.save(ckpt, f'{self.save_dir}/best_macro_acc.pth')

    def run_epoch(self):
        train_acc, train_loss_for_log = self.run_train_epoch()
        valid_acc, valid_loss, balanced_acc, balanced_loss, valid_macro_acc = self.run_valid_epoch()
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
            'valid_acc_balanced': balanced_acc,
            'valid_loss_balanced': balanced_loss,
            'valid_macro_acc': valid_macro_acc,
            **dict(train_loss_for_log),
            **{k:v for k,v in zip(['avail_memory', 'rss_memory'], list(get_mem()))}
            })
            # save angle matrix image each epoch
            self.save_angle_mat(self.epoch)
        self.save(valid_acc, valid_macro_acc)


    def train(self):
        for epoch in range(self.epoch, self.args.n_epochs):
            self.epoch = epoch 
            if self.args.world_size > 1:
                self.train_loader.sampler.set_epoch(epoch)
                self.valid_loader.sampler.set_epoch(epoch)
            self.run_epoch()

        if self.args.world_size == 1 or self.args.rank == 0 :
            import pickle 
            with open(f'{self.save_dir}/log.pkl','wb') as f:
                pickle.dump(self.log,f)
            # Log final angle matrix image to W&B
            try:
                final_img = os.path.join(self.save_dir, 'angle_mat', f"{str(self.epoch).zfill(4)}.png")
                if os.path.exists(final_img):
                    wandb.log({'final/angle_matrix': wandb.Image(final_img), 'final_epoch': self.epoch})
                    # Save 5-second GIF of angle matrix evolution
                    self.save_angle_gif(duration_s=5.0)
                    angle_dir = os.path.join(self.save_dir, 'angle_mat')
                    if os.path.isdir(angle_dir):
                        shutil.make_archive(os.path.join(self.save_dir, 'angle_mat'), 'zip', root_dir=self.save_dir, base_dir='angle_mat')
                        shutil.rmtree(angle_dir)
            except Exception as _e:
                pass

    @torch.inference_mode()
    def save_angle_mat(self, epoch):
        if not (self.args.world_size == 1 or self.args.rank == 0):
            return
        save_dir = os.path.join(self.save_dir, 'angle_mat')
        os.makedirs(save_dir, exist_ok=True)
        kernel = self.model.get_kernel() if self.args.world_size == 1 else self.model.module.get_kernel()
        angle_mat = (torch.arccos((kernel.T@kernel).clamp(-1.0,1.0)) * 180.0 / torch.pi).detach().cpu().numpy()
        plot_angle_matrix(angle_mat, os.path.join(save_dir, f"{str(epoch).zfill(4)}.jpeg"))

    @torch.inference_mode()
    def save_angle_gif(self, duration_s: float = 5.0):
        if not (self.args.world_size == 1 or self.args.rank == 0):
            return
        angle_dir = os.path.join(self.save_dir, 'angle_mat')
        if not os.path.isdir(angle_dir):
            return
        # Collect frames
        image_files = sorted([
            os.path.join(angle_dir, f)
            for f in os.listdir(angle_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        if len(image_files) == 0:
            return
        try:
            import imageio
            images = [imageio.imread(fp) for fp in image_files]
            per_frame = duration_s / max(1, len(images))
            durations = [per_frame] * len(images)
            gif_path = os.path.join(self.save_dir, 'angle_mat.gif')
            imageio.mimsave(gif_path, images, duration=durations)
        except Exception:
            pass


if __name__ == '__main__':
    args = get_args()
    trainer = Trainer(args)
    trainer.train()

