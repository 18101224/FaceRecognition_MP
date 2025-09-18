import torch 
import torch.distributed as dist
from dataset import masking_pair, get_fer_transforms, FER, DistributedSamplerWrapper, ImbalancedDatasetSampler
from aligners import get_aligner
from argparse import ArgumentParser
from torch.distributed import init_process_group, barrier
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm 
from utils import sync_scalar, get_ldmk, get_exp_id, get_acc, crop_to_square_grid
from Loss.Adv import compute_adv_loss, analyze_and_update_gradients
import wandb 
import os 
import math
from opt import SAM
from datetime import timedelta
from models import ImbalancedModel
from torchvision.utils import save_image
from collections import defaultdict
from Loss import KCL, Moco
from models import compute_class_spherical_means,calc_class_mean, slerp
from models import dim_dict
from copy import deepcopy


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    
    tensors_gather = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output



def init_kcl(args, model,dataloader):
    if (args.mean_weight is not None and not os.path.exists(os.path.join(args.mean_weight, 'init_queue.pth'))) :
        mean = compute_class_spherical_means(dataloader, model, device=torch.device('cuda'), num_classes=7, subset=args.moco_k)
        dist.all_reduce(mean, op=dist.ReduceOp.AVG) if args.world_size > 1 else None
        os.makedirs(args.mean_weight, exist_ok=True) if (args.world_size==1 or args.rank==0) else None 
        torch.save(mean, os.path.join(args.mean_weight, 'init_queue.pth')) if (args.world_size==1 or args.rank==0) else None 
        dist.barrier() if args.world_size > 1 else None 

    elif args.mean_weight is not None:
        mean = torch.load(os.path.join(args.mean_weight, 'init_queue.pth'), weights_only=False, map_location=torch.device('cpu'))
    init_queue = mean if args.mean_weight is not None else None
    moco = Moco(args, deepcopy(model).cuda(), num_classes=7, init_queue=init_queue, dim=dim_dict[args.model_type][0])
    kcl = KCL(temperature=args.temperature, include_positives_in_denominator=args.include_positives_in_denominator,
     exclude_same_class_from_negatives=args.exclude_same_class_from_negatives,
    use_batch_negatives=args.use_batch_negatives)
    return moco, kcl, 

def get_optimizer(model, args):
    opt = SAM(model.parameters(),base_optimizer=torch.optim.AdamW,lr=args.learning_rate,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt,gamma=0.98) if args.dataset_name == 'RAF-DB' else torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.n_epochs,eta_min=args.learning_rate/100)
    return opt, scheduler

def get_loaders(args):
    train_transform = get_fer_transforms(train=True)
    valid_transform = get_fer_transforms(train=False)
    train_dataset = FER(args=args, train=True, transform=train_transform, idx=True)
    valid_dataset = FER(args=args, train=False, transform=valid_transform, idx=True)
    train_dataset_wo_aug = FER(args=args, train=True, transform=valid_transform, idx=False, balanced=False)
    if args.world_size > 1 : 
        train_sampler = DistributedSampler(train_dataset, shuffle=True) if not args.use_sampler \
            else DistributedSamplerWrapper(ImbalancedDatasetSampler(train_dataset, labels=train_dataset.labels), shuffle=True)
        valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
        train_sampler_wo_aug = DistributedSampler(train_dataset_wo_aug, shuffle=True)
    else:
        train_sampler  = ImbalancedDatasetSampler(train_dataset, labels=train_dataset.labels) if args.use_sampler else None
        valid_sampler = None
        train_sampler_wo_aug = None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=args.num_workers, pin_memory=True)
    train_loader_wo_aug = DataLoader(train_dataset_wo_aug, batch_size=args.batch_size, sampler=train_sampler_wo_aug,shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return train_loader, valid_loader, train_loader_wo_aug

def get_model(args):
    model =  ImbalancedModel(num_classes=7, model_type=args.model_type, feature_branch=False, 
    feature_module=False, regular_simplex=False, cos=True, decomposition=args.decomposition, learnable_input_dist=False, input_layer=False,
    freeze_backbone=False, remain_backbone=False)
    return model.cuda() if args.world_size ==1 else DDP(model.cuda(), device_ids=[args.local_rank], find_unused_parameters=True)

def get_args():
    args = ArgumentParser()

    # moco args 
    args.add_argument('--moco_k', type=int)
    args.add_argument('--temperature', type=float)
    args.add_argument('--include_positives_in_denominator', default=False)
    args.add_argument('--exclude_same_class_from_negatives', default=False)
    args.add_argument('--use_batch_negatives', default=False)
    
    # model args 
    args.add_argument('--model_type', choices=['ir50', 'kp_rpe'])
    args.add_argument('--mean_weight', default=None,)

    # dataset_args 
    args.add_argument('--dataset_name', choices=['RAF-DB', 'AffectNet'])
    args.add_argument('--dataset_path', type=str)
    args.add_argument('--num_classes', type=int, default=7)

    # research args 
    args.add_argument('--id_strategy', choices=['FR-clustering', 'masking'], default=None)
    args.add_argument('--n_blocks', type=int, default=10)
    args.add_argument('--detach_lowlevel', default=False)
    args.add_argument('--beta', type=float )
    args.add_argument('--partial_update', default=False)
    args.add_argument('--include_backbone', default=False)
    args.add_argument('--anchor_mask', default=False)
    args.add_argument('--decomposition', choices=['Cayley',], default=False)
    args.add_argument('--loss', choices=['CE', 'KCL', 'KBCL'], default='CE')
    args.add_argument('--kcl_k', type=int, default=1)

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
    args.add_argument('--debug',default=False)
    args.add_argument('--measure_grad', default=False)
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
    if (not args.decomposition) and (not args.include_backbone):
        args.measure_grad = False
    return args




class Trainer:
    def __init__(self, args):
        self.args = args 
        self.aligner = get_aligner('checkpoint/adaface_vit_base_kprpe_webface12m').cuda() if args.model_type == 'kp_rpe' or args.id_strategy == 'masking' else None
        self.model = get_model(args)
        self.train_loader, self.valid_loader, self.train_loader_wo_aug = get_loaders(args)
        self.init_logs()
        self.opt, self.scheduler = get_optimizer(self.model, self.args)
        if self.args.loss in ['KCL', 'KBCL']:
            self.moco, self.kcl = init_kcl(self.args, self.model.backbone if self.args.world_size==1 else self.model.module.backbone, self.train_loader_wo_aug)
            mean = compute_class_spherical_means(loader=self.train_loader_wo_aug, model=self.model.backbone if self.args.world_size==1 else \
                self.model.module.backbone, device=torch.device('cuda'), num_classes=7)
            dist.all_reduce(mean, op=dist.ReduceOp.AVG) if self.args.world_size > 1 else None
            self.mean = torch.nn.functional.normalize(mean, dim=1)
            print(self.mean.shape)
            if self.args.world_size==1 or self.args.rank ==0 :
                torch.save(self.mean, os.path.join(self.args.mean_weight, 'class_mean.pth'))
            dist.barrier() if args.world_size > 1 else None 
            to_change = self.model if args.world_size == 1 else self.model.module 
            to_change.weight.data = self.mean.T 

        
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
        self.log = defaultdict(list)
        self.best_acc = 0
        if not os.path.exists(f'checkpoint/{self.id}') and (self.args.world_size == 1 or self.args.rank == 0):
            os.makedirs(f'checkpoint/{self.id}')
        
    def transform(self, img, ldmk):
        if self.args.id_strategy == 'masking':
            anchor_img, neg_img = masking_pair(img, ldmk, n_blocks=self.args.n_blocks, block_size=7, anchor_mask=self.args.anchor_mask)
            
            if self.args.debug:
                anchor_to_save, nrow_a = crop_to_square_grid(anchor_img)
                neg_to_save, nrow_n = crop_to_square_grid(neg_img)
                save_image(anchor_to_save, f'results/anchor_img_{self.args.id_strategy}.png', nrow=nrow_a)
                save_image(neg_to_save, f'results/neg_img_{self.args.id_strategy}.png', nrow=nrow_n)
                import sys; sys.exit()

            return anchor_img, neg_img 
        elif self.args.id_strategy == 'FR-clustering':
            raise NotImplementedError

    # def measure_loss(self,model, l1, l2):
    #     dict_grad_norms  ={}
    #     dict_cosine_sims = {}
    #     grad_backup = {}

    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             if param.grad is not None:
    #                 grad_backup[name] = param.grad.clone()
    #             param.grad = None 
            
    #     l1.backward(retain_graph=True)
    #     grad_loss1 = {}
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             grad_loss1[name] = param.grad.clone() if param.grad is not None else None
        
    #     for

    def backp(self, ce_loss, fr_loss):
        to_backp = self.model.module if self.args.world_size > 1 else self.model
        if not self.args.measure_grad:
            ce_loss.backward(retain_graph=True)
            if  self.args.decomposition :
                torch.autograd.backward(tensors=[fr_loss*self.args.beta],
                inputs=(list(to_backp.backbone.body3.parameters())+list(to_backp.decomposition.parameters())) if self.args.partial_update \
                    else list(to_backp.parameters()))
            return None, None, None
        else:
            cos_sims, norms_ce, norms_fr = analyze_and_update_gradients(to_backp, ce_loss, fr_loss, self.opt, include_backbone=self.args.partial_update, world_size=self.args.world_size)
            return cos_sims, norms_ce, norms_fr


    def run_train_forward(self, img, label):
        img = img.detach().cuda() # masking 등이 없는 img 임. 
        label = label.detach().cuda()
        bs = label.shape[0]
        ldmk = get_ldmk(img, self.aligner) if self.aligner is not None else None
        if self.args.loss not in ['KCL', 'KBCL']:
            anchor_img, neg_img = self.transform(img, ldmk) 
            entire = torch.concat([anchor_img, neg_img], dim=0)
            entire_ldmk = ldmk.repeat(2,1,1) if (ldmk is not None and self.args.model_type == 'kp_rpe') else None
            logits, fer_features, fr_features = self.model(entire, keypoint=entire_ldmk, features=True)
            anchor_fr_features, neg_fr_features = torch.split(fr_features, [bs,bs],dim=0)
            logits, _ = torch.split(logits, [bs,bs],dim=0)
            ce_loss = torch.nn.functional.cross_entropy(logits,label)
            fr_loss = compute_adv_loss(anchor_fr_features, neg_fr_features)
        else:
            logits, features, centers = self.model(img, keypoint=ldmk, features=True)
            features_to_mean = concat_all_gather(features) if self.args.world_size > 1 else features
            labels_to_mean = concat_all_gather(label) if self.args.world_size > 1 else label
            temp_mean, temp_mask = calc_class_mean(features_to_mean, labels_to_mean, self.args.num_classes)
            temp_mean = slerp(self.mean[temp_mask==True].detach(), temp_mean[temp_mask==True], 0.001)
            now_mean = self.mean.clone().detach()
            now_mean[temp_mask==True] = temp_mean 
            self.mean[temp_mask==True] = now_mean[temp_mask==True]
            ce_loss = torch.nn.functional.cross_entropy(logits,label)
            features = torch.cat([features, now_mean, centers], dim=0)
            label = torch.cat( [label, torch.arange(self.args.num_classes, device=torch.device('cuda')).repeat(2)] , dim=0)
            k_features, k_labels = self.moco.get_k(label, self.args.kcl_k)
            fr_loss = self.kcl(features, label, k_features, k_labels)
            # add weight and class mean 

            
        return ce_loss, self.args.beta*fr_loss, logits 
    
    @torch.no_grad()
    def run_valid_forward(self, img, label):
        img = img.cuda()
        label = label.cuda()
        ldmk = get_ldmk(img, self.aligner) if (self.aligner is not None and self.args.model_type == 'kp_rpe') else None
        logits = self.model(img, keypoint=ldmk, features=False)
        loss = torch.nn.functional.cross_entropy(logits,label)
        return loss, logits 

    def add_log(self, cos_sims, norms_ce, norms_fr):
        if self.args.include_backbone: 
            self.log['cos_sim_backbone'].append(cos_sims['backbone.body3'])
            self.log['norms_ce_backbone'].append(norms_ce['backbone.body3'])
            self.log['norms_fr_backbone'].append(norms_fr['backbone.body3'])
        
        self.log['cos_sim_decomposition'].append(cos_sims['decomposition'])
        self.log['norms_ce_decomposition'].append(norms_ce['decomposition'])
        self.log['norms_fr_decomposition'].append(norms_fr['decomposition'])

    def run_train_epoch(self):
        self.model.train()
        total_ce_loss = 0 
        total_fr_loss = 0 
        total_acc = 0 

        for img, label, idx in tqdm(self.train_loader, disable=self.args.world_size > 1 and self.args.rank != 0):
            self.model.zero_grad()

            ce_loss, fr_loss, logits = self.run_train_forward(img,label)
            if self.args.loss in ['KCL', 'KBCL']:
                (ce_loss+fr_loss).backward()
                self.opt.first_step(zero_grad=True)
            else:
                self.backp(ce_loss, fr_loss)
                self.opt.first_step(zero_grad=True)
                
            ce_loss, fr_loss, logits = self.run_train_forward(img,label)
            if self.args.loss in ['KCL', 'KBCL']:
                (ce_loss+fr_loss).backward()
                self.opt.second_step(zero_grad=True)
                self.moco.momentum_update(self.model.backbone if self.args.world_size==1 else self.model.module.backbone)
                self.moco.enqueue(img.cuda(),label.cuda(),ldmks=None)
            else:
                cos_sims, norms_ce, norms_fr = self.backp(ce_loss, fr_loss)
                if self.args.measure_grad and (args.world_size > 1 or args.rank == 0):
                    self.add_log(cos_sims, norms_ce, norms_fr)
                self.opt.second_step(zero_grad=True)
            bs = label.shape[0]
            total_ce_loss += ce_loss.detach().item() * bs
            total_fr_loss += fr_loss.detach().item() * bs
            total_acc += get_acc(logits, label.cuda()) * bs
        

        if self.args.world_size > 1 :
            total_ce_loss = sync_scalar(torch.tensor(total_ce_loss, device=torch.device('cuda')))
            total_fr_loss = sync_scalar(torch.tensor(total_fr_loss, device=torch.device('cuda')))
            total_acc = sync_scalar(torch.tensor(total_acc, device=torch.device('cuda')))
        
        if self.args.measure_grad and (args.world_size > 1 or args.rank == 0):
            if self.args.include_backbone:
                self.log['epoch_cos_sim_backbone'] = sum(self.log['cos_sim_backbone'][-len(self.train_loader):]) / len(self.train_loader)
                self.log['epoch_norms_ce_backbone'] = sum(self.log['norms_ce_backbone'][-len(self.train_loader):]) / len(self.train_loader)
                self.log['epoch_norms_fr_backbone'] = sum(self.log['norms_fr_backbone'][-len(self.train_loader):]) / len(self.train_loader)
            self.log['epoch_cos_sim_decomposition'] = sum(self.log['cos_sim_decomposition'][-len(self.train_loader):]) / len(self.train_loader)
            self.log['epoch_norms_ce_decomposition'] = sum(self.log['norms_ce_decomposition'][-len(self.train_loader):]) / len(self.train_loader)
            self.log['epoch_norms_fr_decomposition'] = sum(self.log['norms_fr_decomposition'][-len(self.train_loader):]) / len(self.train_loader)
            if self.args.include_backbone:
                print(f'epoch_cos_sim_backbone: {self.log["epoch_cos_sim_backbone"]:.4f}, epoch_norms_ce_backbone: {self.log["epoch_norms_ce_backbone"]:.4f}, epoch_norms_fr_backbone: {self.log["epoch_norms_fr_backbone"]:.4f}, epoch_cos_sim_decomposition: {self.log["epoch_cos_sim_decomposition"]:.4f}, epoch_norms_ce_decomposition: {self.log["epoch_norms_ce_decomposition"]:.4f}, epoch_norms_fr_decomposition: {self.log["epoch_norms_fr_decomposition"]:.4f}')
            else:
                print(f'epoch_cos_sim_decomposition: {self.log["epoch_cos_sim_decomposition"]:.4f}, epoch_norms_ce_decomposition: {self.log["epoch_norms_ce_decomposition"]:.4f}, epoch_norms_fr_decomposition: {self.log["epoch_norms_fr_decomposition"]:.4f}')
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
                'best_acc': self.best_acc,
                'epoch_cos_sim_backbone': self.log['epoch_cos_sim_backbone'],
                'epoch_norms_ce_backbone': self.log['epoch_norms_ce_backbone'],
                'epoch_norms_fr_backbone': self.log['epoch_norms_fr_backbone'],
                'epoch_cos_sim_decomposition': self.log['epoch_cos_sim_decomposition'],
                'epoch_norms_ce_decomposition': self.log['epoch_norms_ce_decomposition'],
                'epoch_norms_fr_decomposition': self.log['epoch_norms_fr_decomposition'],
            })
        print(f'train_acc: {train_acc:.4f}, valid_acc: {valid_acc:.4f}, best_acc: {self.best_acc:.4f}')
        self.save(valid_acc)

    def train(self):
        for epoch in range(self.args.n_epochs):
            self.epoch = epoch 
            self.run_epoch()
        if self.args.world_size == 1 or self.args.rank == 0:
            import pickle 
            with open(f'{self.save_dir}/log.pkl','wb') as f:
                pickle.dump(self.log,f)

if __name__ == '__main__':
    args = get_args()
    trainer = Trainer(args)
    trainer.train()