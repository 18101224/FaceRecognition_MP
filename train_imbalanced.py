import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
from tqdm import tqdm
import wandb
import os
from argparse import ArgumentParser
from models import ImbalancedModel
from collections import Counter
from torch.optim import SGD
from Loss.Imbalanced import get_angle_loss, BCLLoss, weight_scheduling, ECELoss
from dataset import get_cifar_dataset, get_transform, Large_dataset, FER, ImbalancedDatasetSampler, DistributedSamplerWrapper
from utils import get_exp_id
from analysis import plot_angle_matrix
import numpy as np
from utils.plot import plot_angle_gif
from collections import defaultdict
from copy import deepcopy
import shutil, json 
from torch import distributed as dist
from opt import adjust_learning_rate, get_scheduler, get_optimizer, SAM 
import time 
from torch.profiler import profile, schedule, ProfilerActivity
import random

# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# torch.cuda.manual_seed_all(42)  
    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def _jsonify(obj):
    """Convert objects to JSON-serializable forms recursively."""
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonify(v) for v in obj]
    try:
        import numpy as np  # local import in case numpy is unavailable in some contexts
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if hasattr(obj, '__dict__'):
        return _jsonify(vars(obj))
    return str(obj)

def profile_train_if_enabled(func):
    def wrapper(self, *args, **kwargs):
        # Only profile if enabled via args; otherwise run normally
        if not getattr(self.args, 'use_profiler', False):
            return func(self, *args, **kwargs)
        # Ensure profiler exists (created in __init__) and wrap the call
        with self.profiler as _prof:
            result = func(self, *args, **kwargs)
            try:
                # Print a brief summary to stdout for quick inspection
                print(_prof.key_averages().table(sort_by="self_cpu_time_total"))
            except Exception:
                pass
            return result
    return wrapper

def get_dataset(args, train:bool):
    dataset_name = args.dataset_name
    if 'cifar' in dataset_name:
        transform = get_transform(args=args, train=train)
        result = get_cifar_dataset(dataset_name=args.dataset_name,root=args.dataset_path,train=train, imb_factor=args.imb_factor, imb_type='exp',transform=transform)
    elif 'imagenet_lt' == dataset_name:
        transform = get_transform(args,train=train)
        result = Large_dataset(root=args.dataset_path, train=train, transform=transform)
    elif 'RAF-DB' == dataset_name or 'AffectNet' == dataset_name:
        transform = get_transform(args,train=train)
        result = FER(args=args, train=train, transform=transform, idx=False)
    else:
        raise ValueError(f'Dataset {args.dataset_name} not supported')
    return result 

    
def get_model(args):
    if 'cifar' in args.dataset_name:
        n_c = 100 if '100' in args.dataset_name else 10
        model = ImbalancedModel(num_classes=n_c, model_type=args.model_type, feature_module=args.feature_module, feature_branch=args.feature_branch)
        return model
    elif 'imagenet_lt' == args.dataset_name:
        model = ImbalancedModel(num_classes=1000, model_type=args.model_type)
        return model
    elif 'inat' == args.dataset_name:
        model = ImbalancedModel(num_classes=8142, model_type=args.model_type)
        return model
    elif 'RAF-DB' == args.dataset_name or 'AffectNet' == args.dataset_name:
        model = ImbalancedModel(num_classes=7, model_type=args.model_type, feature_module=args.feature_module, feature_branch=args.feature_branch)
        return model
    else:
        raise ValueError(f'Dataset {args.dataset_name} not supported')


def get_args():
    args = ArgumentParser()
    # Training hyperparameters
    args.add_argument('--resume_path', type=str, default=None)
    args.add_argument('--gamma', type=float, default=0.1)
    args.add_argument('--learning_rate', type=float, default=0.1)
    args.add_argument('--batch_size', type=int, default=128)
    args.add_argument('--n_epochs', type=int, default=200)
    args.add_argument('--scheduler', type=str, choices=['cosine', 'warmup'], default='cosine')
    args.add_argument('--weight_decay', type=float, default=1e-4)

    args.add_argument('--randaug_n', type=int, default=2, help='Number of RandAugment transformations to apply')
    args.add_argument('--randaug_m', type=int, default=10, help='Magnitude for RandAugment transformations')
    args.add_argument('--warmup_epochs', type=int, default=5)
    args.add_argument('--cos', default=False)
    args.add_argument('--momentum', type=float, default=0.9)

    args.add_argument('--feature_module', help='deepcomplex_depth', default=False)
    # Distributed training
    args.add_argument('--world_size', type=int, default=1)
    args.add_argument('--local_rank', type=int, default=None)
    args.add_argument('--rank', type=int, default=None)
    args.add_argument('--use_tf', default=False)
    args.add_argument('--num_workers', type=int, default=0)
    args.add_argument('--use_profiler', default=False)
    

    # Logging and saving
    args.add_argument('--use_wandb', default=False)


    # Research hyperparameters
    args.add_argument('--cosine_scaling',type=float,default=1.0)
    args.add_argument('--cosine_constant_margin', type=float, default=0.0,
                     help='Constant margin for cosine similarity loss')
    args.add_argument('--angle_loss', type=float, default=False)
    args.add_argument('--loss', default='CE', help="CE_ECE_BCL")
    args.add_argument('--ce_weight', type=float, default=1.0)
    args.add_argument('--cl_weight', type=float, default=1.0)
    args.add_argument('--ece_weight', type=float, default=1.0)
    args.add_argument('--ece_scheduling', default=False, choices=['linear','cosine','sigmoid','piecewise'])
    args.add_argument('--temperature', type=float, default=0.1)
    args.add_argument('--regular_simplex', default=False)
    args.add_argument('--splitted_contrastive_learning', default=False)
    args.add_argument('--use_mean', default=False)
    args.add_argument('--surrogate', default=False)
    args.add_argument('--k',type=int)
    args.add_argument('--hard_weight', type=float, default=1)
    args.add_argument('--soft_weight', type=float, default=1)
    # Model checkpoint
    args.add_argument('--model_type', type=str, choices=['resnet32','resnet50','resnext50','ir50','e2_resnet32','e2_resnext50'], default='resnet32')
    args.add_argument('--feature_branch', default=False)
    args.add_argument('--use_sampler',default=False)
    #dataset information 
    args.add_argument('--dataset_name',type=str,choices=['cifar100','svhn','cifar10','imagenet_lt','inat','RAF-DB','AffectNet'])
    args.add_argument('--dataset_path',type=str)
    args.add_argument('--download', default=False)
    args.add_argument('--imb_factor', type=float,)
    args.add_argument('--imb_type', type=str, choices=['exp','step'], default='exp')
    args.add_argument('--aug', default=False)
    args.add_argument('--cutout', default=False)
    
    args = args.parse_args() 

    vars(args)['server'] = os.getenv('SERVER','0')
    # If resuming, load checkpoint args and allow GPU settings to be overridden
    resume_ckpt = None
    if args.resume_path is not None:
        ckpt_path = os.path.join(args.resume_path, 'latest.pth')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        ckpt_args = ckpt['args']
        # Overwrite GPU-related args with current command line if provided
        for gpu_arg in ['local_rank', 'rank', 'world_size']:
            val = getattr(args, gpu_arg, None)
            if val is not None:
                setattr(ckpt_args, gpu_arg, val)
        ckpt_args.resume_path = args.resume_path  # keep resume_path for Trainer
        args = ckpt_args
        args._resumed_ckpt = ckpt_path  # mark for Trainer
        resume_ckpt = ckpt
    else:
        args._resumed_ckpt = None
    
    if args.world_size > 1:
        init_process_group('nccl')
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.rank = int(os.environ['RANK'])
        args.batch_size = args.batch_size // args.world_size
        torch.cuda.set_device(args.local_rank)
    else:
        torch.cuda.set_device(0)
    
    if args.model_type == 'softmax' and (args.cosine_constant_margin!=0.0 or args.cosine_scaling!=1.0):
        raise ValueError('Cosine constant margin is not supported for softmax model')
    
    if args.use_tf : 
        torch.backends.cuda.matmul.allow_tf32 = True 
        torch.backends.cudnn.allow_tf32 = True 
        torch.backends.cudnn.benchmark = True 
    return args

def include(loss, loss_names):
    losses = list(loss.split('_'))
    for loss in losses : 
        for loss_name in loss_names : 
            if loss_name == loss : 
                return True 
    return False 

def get_losses(loss):
    losses = list(loss.split('_'))
    if 'CE' not in losses:
        losses.append('CE')
    result = {}
    for loss in losses:
        result[loss] = []
    return result 

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = args.local_rank if args.world_size > 1 else torch.device('cuda')
        self._start_epoch = 0
        self._resumed_ckpt = getattr(args, '_resumed_ckpt', None)
        self._resume_ckpt_data = None
        if self._resumed_ckpt:
            # Load checkpoint data for later use
            self._resume_ckpt_data = torch.load(self._resumed_ckpt, map_location='cpu')
            self.wandb_run_id = self._resume_ckpt_data.get('wandb_run_id', None)
        else:
            self.wandb_run_id = None
        
        ##################################################
        #### 🚀 DATASETS & DATALOADERS INITIALIZATION ####
        ##################################################
        # Initialize datasets and dataloaders

        self.train_dataset = get_dataset(args, train=True)
        self.test_dataset = get_dataset(args, train=False)
        
        # Create samplers for distributed training
        if args.world_size > 1 :
            if args.use_sampler : 
                self.train_sampler = DistributedSamplerWrapper(ImbalancedDatasetSampler(self.train_dataset, labels=self.train_dataset.labels),shuffle=True)
            else: 
                self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
        else:
            if args.use_sampler : 
                self.train_sampler = ImbalancedDatasetSampler(dataset=self.train_dataset,labels=self.train_dataset.labels, shuffle=True)
            else:
                self.train_sampler = None
        self.test_sampler = DistributedSampler(self.test_dataset, shuffle=False) if args.world_size > 1 else None
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            num_workers=self.args.num_workers,
            persistent_workers=(self.args.num_workers > 1),
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=self.test_sampler,
            num_workers=self.args.num_workers,
            persistent_workers=(self.args.num_workers > 1),
            pin_memory=True
        )
        

        #############################################
        #### 🧠 MODEL & OPTIMIZER INITIALIZATION ####
        #############################################
        # Initialize model
        
        self.model = get_model(args)
        self.model = self.model.to(self.device)
        
        if args.world_size > 1:
            self.model = DDP(self.model, device_ids=[args.local_rank],find_unused_parameters=True)
        
        # Initialize optimizer and scheduler
        self.optimizer = get_optimizer(args, self.model)
        self.scheduler = get_scheduler(args, self.optimizer) # none or 
        # Restore model/optimizer/scheduler/logs if resuming
        if self._resume_ckpt_data is not None:
            state = self._resume_ckpt_data
            if args.world_size > 1:
                self.model.module.load_state_dict(state['model_state_dict'])
            else:
                self.model.load_state_dict(state['model_state_dict'])
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(state['scheduler_state_dict'])
            self.best_acc = state.get('best_acc', 0)
            self.best_macro_acc = state.get('best_macro_acc', 0)
            self.log = state.get('log', self.log)
            self._start_epoch = state.get('epoch', 0) + 1  # resume from next epoch


        ##################################################
        ######### 📊 Loss settings #######################
        ##################################################
        if include(self.args.loss, ['BCL']) :
            self.bcl = BCLLoss(cls_num_list=self.train_dataset.img_num_list, args=self.args, temperature=self.args.temperature)
        if include(self.args.loss, ['ECE']) :
            self.ece = ECELoss(args=self.args, k=self.args.k, hard_weight=self.args.hard_weight, soft_weight=self.args.soft_weight, num_classes=np.max(self.train_dataset.labels)+1, surrogate=self.args.surrogate)
        #############################################
        #### 📊 METRICS & LOGGING INITIALIZATION ####
        #############################################
        self.best_acc = 0
        self.best_macro_acc = 0
        losses = get_losses(self.args.loss)
        self.log = {
            'train_acc': [], 'train_loss': [],
            'test_acc': [], 'test_loss': [],
            'train_macro_acc': [], 'test_macro_acc': [],
            'angle': [],
            **losses
        }
        if args.ece_scheduling :
            self.ece_original = deepcopy(self.args.ece_weight)
        # Initialize wandb
        if args.world_size == 1 or args.rank == 0:
            if self._resume_ckpt_data is not None and 'id' in self._resume_ckpt_data:
                self.id = self._resume_ckpt_data['id']
            else:
                self.id = get_exp_id(args)
            os.makedirs(f'checkpoint/{self.id}', exist_ok=True)
            os.makedirs(f'checkpoint/{self.id}/angle_mat', exist_ok=True)
        if args.world_size > 1:
            # Broadcast id from rank 0 to all other ranks
            import torch.distributed as dist
            if args.rank == 0:
                id_bytes = self.id.encode('utf-8')
                id_tensor = torch.ByteTensor(list(id_bytes)).to(self.device)
                length_tensor = torch.tensor([len(id_bytes)], device=self.device)
            else:
                id_tensor = torch.ByteTensor(256).to(self.device)  # max id length 256
                length_tensor = torch.tensor([0], device=self.device)
            # Broadcast length first
            dist.broadcast(length_tensor, src=0)
            # Broadcast id bytes
            if args.rank != 0:
                id_tensor = id_tensor[:length_tensor.item()]
            dist.broadcast(id_tensor, src=0)
            if args.rank != 0:
                self.id = bytes(id_tensor.cpu().numpy().tolist()).decode('utf-8') 
        
        if (args.world_size == 1 or args.rank == 0) and args.use_wandb :
            self._init_wandb()
        
        class_dist = Counter(self.train_dataset.labels)
        self.class_dist = [0]*len(class_dist)
        for key, value in class_dist.items():
            self.class_dist[key] = value
        self.class_dist = torch.tensor(self.class_dist,device=self.device)

        val_class_dist = Counter(self.test_dataset.labels)
        self.validation_class_dist = [0]*len(val_class_dist)
        for key, value in val_class_dist.items():
            self.validation_class_dist[key] = value
        self.validation_class_dist = torch.tensor(self.validation_class_dist,device=self.device)
        self.loss_names = ['CE', 'ECE','BCL']
        if self.args.use_profiler : 
            self.profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(f'checkpoint/{self.id}/tb_logs'),
                schedule=schedule(wait=1, warmup=1, active=3, repeat=2),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
                with_modules=True,
            )
        print(f'num_classes: {len(self.validation_class_dist)}')

    def _init_wandb(self):
        wandb.login()
        import uuid 
        self.wandb_id = str(uuid.uuid4())
        wandb.init(
            project=f'{self.args.dataset_name} Training',
            name=self.id,
            config=dict(vars(self.args)),
            id=self.wandb_id,
            resume='allow',
        )
    
    def run_train_forward(self, images, labels, train=True):
        """Run forward pass for training with optional margin and angle loss."""
        processed_feat, outputs, centers = self.model(images, features=True) # projected_features, logits, projected_centers
        original_outputs = deepcopy(outputs.detach())
        losses = []
        losses_for_log = defaultdict(int)
        if train : 
            if 'BCL' in self.args.loss : 
                _ , f1, f2 = torch.split(processed_feat, [labels.shape[0]]*3, dim=0)
                processed_feat = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                outputs, _, _ = torch.split(outputs, [labels.shape[0]]*3, dim=0)
                loss, bcl = self.bcl(centers=centers, logits=outputs, features=processed_feat,
                                      targets=labels, processed_features=None)
                losses.append(bcl*self.args.cl_weight)
                losses_for_log['BCL'] = bcl.detach().cpu().item() # weight 적용 전 
            
            if include(self.args.loss, ['CE']) and not include(self.args.loss, ['BCL']): 
                loss = torch.nn.functional.cross_entropy(outputs*self.args.cosine_scaling, labels)
            losses.append(loss*self.args.ce_weight)
            losses_for_log['CE'] = loss.detach().cpu().item()

            # Add angle loss if specified
            if include(self.args.loss, ['ECE']) : 
                mode = self.model.module if self.args.world_size > 1 else self.model
                kernel = mode.get_kernel()
                ece = self.ece(kernel.T)
                losses.append(ece)
                losses_for_log['ECE'] = ece.detach().cpu().item()
        else:
            losses = torch.nn.functional.cross_entropy(outputs, labels)
        return losses, original_outputs, outputs, losses_for_log
    
    def train_epoch(self):
        self.model.train()
        total_losses = [0]*10
        total_acc = 0
        total_macro_accs = torch.zeros((len(self.class_dist)),device=self.device)
        losses_for_log = defaultdict(list)

        for batch in tqdm(self.train_loader, disable=self.args.world_size > 1 and self.args.rank != 0):
            if 'BCL' in self.args.loss : 
                images,  labels = batch 
                images = torch.cat(images, dim=0)
            else:
                images, labels = batch
                if isinstance(images, list):
                    images = images[0]
            images, labels = images.to(self.device), labels.to(self.device)
            batch_size = labels.shape[0]
            # First forward-backward pass
            self.optimizer.zero_grad()
            losses, original_outputs, outputs, temp_losses = self.run_train_forward(images, labels)
            loss = sum(losses)
            loss.backward()
            if isinstance(self.optimizer, SAM):
                self.optimizer.first_step(zero_grad=True)
                losses, original_outputs, outputs, temp_losses = self.run_train_forward(images, labels, train=True)
                loss = sum(losses)
                loss.backward()
                self.optimizer.second_step(zero_grad=True)
            else:
                self.optimizer.step()
            # Calculate metrics
            with torch.no_grad():
                for i in range(len(losses)):
                    if losses[i] != 0 : 
                        total_losses[i] += losses[i].item()*batch_size
                if 'BCL' in self.args.loss : 
                    original_outputs = original_outputs[:labels.shape[0]]
                total_acc += self._get_acc(original_outputs, labels) * batch_size
                total_macro_accs += self._get_macro_acc(original_outputs, labels)
                for key, value in temp_losses.items():
                    losses_for_log[key].append(value)
            if getattr(self.args, 'use_profiler', False):
                # Advance profiler scheduling per iteration
                self.profiler.step()

        for key, value in losses_for_log.items():
            mean = np.mean(losses_for_log[key])
            self.log[key].append(mean)
            if self.args.world_size > 1:
                self.log[key][-1] = self._sync_tensor(torch.tensor(self.log[key][-1], device=self.device))
        # Average metrics
        total_samples = len(self.train_dataset)

        avg_acc = total_acc / total_samples
        avg_macro_acc = torch.mean(total_macro_accs.reshape(1,-1) / self.class_dist.reshape(1,-1)).detach().cpu().item()

        # Synchronize metrics across processes
        if self.args.world_size > 1:
            avg_acc = self._sync_tensor(torch.tensor(avg_acc, device=self.device)).item()
            avg_macro_acc = self._sync_tensor(torch.tensor(avg_macro_acc, device=self.device)).item()

        return avg_acc, avg_macro_acc
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        total_acc = 0
        total_macro_accs = torch.zeros((len(self.validation_class_dist)),device=self.device)
        
        for images, labels in tqdm(self.test_loader, disable=self.args.world_size > 1 and self.args.rank != 0):
            if isinstance(images, list):
                images = images[0]
            images, labels = images.to(self.device), labels.to(self.device)
            batch_size = images.size(0)
            # Forward pass
            loss, _, outputs, _ = self.run_train_forward(images, labels, train=False)
            # Calculate metrics
            total_loss += loss.item() * batch_size
            total_acc += self._get_acc(outputs, labels) * batch_size
            total_macro_accs += self._get_macro_acc(outputs, labels) 
        
        # Average metrics
        total_samples = len(self.test_dataset)
        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples
        avg_macro_acc = torch.mean(total_macro_accs.reshape(1,-1) / self.validation_class_dist.reshape(1,-1)).detach().cpu().item()
        
        # Synchronize metrics across processes
        if self.args.world_size > 1 :
            avg_loss = self._sync_tensor(torch.tensor(avg_loss, device=self.device)).item()
            avg_acc = self._sync_tensor(torch.tensor(avg_acc, device=self.device)).item()
            avg_macro_acc = self._sync_tensor(torch.tensor(avg_macro_acc, device=self.device)).item() # should be modified.
        return avg_loss, avg_acc, avg_macro_acc
    
    def _get_acc(self, preds, labels):
        """Calculate accuracy"""
        return (preds.argmax(dim=1) == labels).float().mean().item()
    
    def _get_macro_acc(self, preds, labels):
        num_classes = preds.shape[-1]
        """Calculate macro-averaged accuracy per class"""
        accs = torch.zeros(len(self.class_dist), device=self.device)
        binaries = torch.argmax(preds,dim=-1) == labels
        for c in range(len(self.class_dist)):
            accs[c] += binaries[labels == c].sum()
        return accs
    
    def _sync_tensor(self, tensor):
        """Synchronize tensor across all processes"""
        if self.args.world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor
    
    @torch.no_grad()
    def save_angle_mat(self, epoch):
        mode = self.model.module if self.args.world_size > 1 else self.model
        kernel = mode.get_kernel().detach().cpu().numpy() # dim, num class
        angles = np.arccos(kernel.T@kernel)*180.0/np.pi
        plot_angle_matrix(angles, f'checkpoint/{self.id}/angle_mat/{str(epoch).zfill(4)}.png', dataset_name=self.args.dataset_name)


    def save_checkpoint(self, epoch, is_best=False, is_best_macro=False):
        if self.args.world_size > 1 and self.args.rank != 0:
            return
        
        save_dir = f'checkpoint/{self.id}'  

        # Get model state dict
        model_state = self.model.module.state_dict() if self.args.world_size > 1 else self.model.state_dict()
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'best_acc': self.best_acc,
            'best_macro_acc': self.best_macro_acc,
            'log': self.log,
            'args': self.args,
            'id': self.id,
            'wandb_run_id': getattr(self, 'wandb_run_id', None),
            'feature_module': self.args.feature_module,
            'regular_simplex': getattr(self.args, 'regular_simplex', False),
        }
        

        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, f'{save_dir}/best_acc.pth')
        
        # Save best macro accuracy checkpoint
        if is_best_macro:
            torch.save(checkpoint, f'{save_dir}/best_macro_acc.pth')
        
        torch.save(checkpoint, f'{save_dir}/latest.pth')
        # Save periodic checkpoint
    
    @profile_train_if_enabled
    def train(self):
        start_epoch = getattr(self, '_start_epoch', 0)
        for epoch in tqdm(range(start_epoch, self.args.n_epochs), disable=self.args.world_size > 1 and self.args.rank != 0):
            temp_lr = adjust_learning_rate(self.optimizer, epoch, self.scheduler, self.args)
            if self.args.world_size > 1:
                self.train_sampler.set_epoch(epoch)
            if self.args.ece_scheduling:
                self.args.ece_weight = weight_scheduling(method=self.args.ece_scheduling, beta=self.ece_original, epoch=epoch, n_epochs=self.args.n_epochs)
            # Training
            train_acc, train_macro_acc = self.train_epoch()
            
            # Evaluation
            test_loss, test_acc, test_macro_acc = self.evaluate()
            
            # Update learning rate

            
            # Update best metrics
            is_best = test_acc > self.best_acc
            is_best_macro = test_macro_acc > self.best_macro_acc
            if is_best:
                self.best_acc = test_acc
            if is_best_macro:
                self.best_macro_acc = test_macro_acc
            
            # Update logs

            self.log['train_acc'].append(train_acc)
            self.log['train_macro_acc'].append(train_macro_acc)
            self.log['test_acc'].append(test_acc)
            self.log['test_macro_acc'].append(test_macro_acc)
            # Save checkpoint
            self.save_checkpoint(epoch, is_best, is_best_macro)
            
            # Log to wandb
            if self.args.world_size == 1 or self.args.rank == 0 :
                metrics = {
                    'train_acc': train_acc,
                    'train_macro_acc': train_macro_acc,
                    'test_loss': test_loss,
                    'test_acc': test_acc,
                    'test_macro_acc': test_macro_acc,
                    'best_acc': self.best_acc,
                    'best_macro_acc': self.best_macro_acc,
                    'ece_weight': self.args.ece_weight,
                    'lr': temp_lr,
                }
                if self.args.use_wandb : 
                    wandb.log(metrics)

                print(f'Epoch {epoch}: '
                    f'Train Acc: {train_acc:.4f}, Train Macro Acc: {train_macro_acc:.4f}, '
                    f'Test Acc: {test_acc:.4f}, Test Macro Acc: {test_macro_acc:.4f}')
                if self.args.cos : 
                    self.save_angle_mat(epoch)


        if (self.args.world_size == 1 or self.args.rank == 0) and self.args.cos :
            plot_angle_gif(f'checkpoint/{self.id}/angle_mat', duration=6, fps=30)
            shutil.make_archive(f'checkpoint/{self.id}/angle_mat', 'zip', f'checkpoint/{self.id}/angle_mat')
            img_path = f'checkpoint/{self.id}/angle_mat/{str(epoch).zfill(4)}.png'
            if self.args.use_wandb : 
                wandb.log({'angle_mat': wandb.Image(img_path)})
            time.sleep(30)
            with open(f'checkpoint/{self.id}/args.json','w') as f :
                json.dump(_jsonify(vars(self.args)), f, indent=2)  

        return self.best_acc, self.best_macro_acc, self.id 
    
if __name__ == '__main__':
    args = get_args()
    trainer = Trainer(args)
    best_acc, best_macro_acc, id = trainer.train()
    if (args.world_size==1 or args.rank==0) and os.path.exists(f'checkpoint/{id}/angle_mat'):
        shutil.rmtree(f'checkpoint/{id}/angle_mat',ignore_errors=True)
