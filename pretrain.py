import torch.optim.lr_scheduler

from dataset import FER
from models import kprpe_fer, load_kprpe_finetuned
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import get_acc, sync, get_macro_acc, sync_tensor
from utils.fer_logger import init_wandb
from argparse import ArgumentParser
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from dataset import ImbalancedDatasetSampler
from kp import get_aligner
from opt import *
import wandb,os,pickle,aligners
from copy import deepcopy
from Loss import including_margin, get_label_noise, get_angle_loss
from torch import nn

def get_args():
    args = ArgumentParser()
    # for logs
    args.add_argument('--server',required=True,help='server name')
    args.add_argument('--wandb_token')
    args.add_argument('--name',default=False)
    args.add_argument('--save_every',default=False)

    # for checkpoint
    args.add_argument('--use_hf',default=False)
    args.add_argument('--token_path',default=None)
    args.add_argument('--instance_adaloss_ckpt')
    args.add_argument('--force_download',default=False)
    args.add_argument('--quality_model_path')


    # training hyperparameteres
    args.add_argument('--learning_rate',type=float)
    args.add_argument('--batch_size',type=int)
    args.add_argument('--kp_rpe_cfg_path')
    args.add_argument('--ckpt')
    args.add_argument('--n_epochs', type=int)

    # for dataset
    args.add_argument('--dataset_path')
    args.add_argument('--dataset_name')

    # class balancing
    args.add_argument('--ldam_weight',type=float)
    args.add_argument('--class_quality_loss',default=False,help='class quality loss')
    args.add_argument('--class_ada_loss',default=False,help='class proportion')
    args.add_argument('--proportion_alpha',type=float)
    args.add_argument('--quality_beta',type=float)
    args.add_argument('--class_alpha',type=float)
    args.add_argument('--instance_ada_loss',default=False)
    args.add_argument('--angle_loss',default=False, type=float)

    # instance adaptive
    args.add_argument('--cos_margin_loss',default=False,help='class angular margin')
    args.add_argument('--cos_constant_margin',type=float)
    args.add_argument('--margin',type=float)

    # distributed training
    args.add_argument('--world_size',default=1,type=int)
    args.add_argument('--local_rank')
    args.add_argument('--rank')
    args.add_argument('--instance_ada_dropout',default=False)
    args.add_argument('--instance_alpha',type=float)


    args = args.parse_args()
    if args.world_size > 1 :
        init_process_group('nccl',world_size=args.world_size)
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.rank = int(os.environ['RANK'])
        args.batch_size = args.batch_size // args.world_size
        torch.cuda.set_device(args.local_rank)
    else:
        torch.cuda.set_device(0)
    return args


class PreTrainer :
    def __init__(self,args):
        self.train_set = FER(args,args.quality_model_path)
        self.valid_set = FER(args,args.quality_model_path,train=False)
        self.train_sampler = ImbalancedDatasetSampler(self.train_set,labels=self.train_set.labels)
        if args.world_size > 1 :
            from catalyst.data import DistributedSamplerWrapper
            self.train_sampler = DistributedSamplerWrapper(self.train_sampler,shuffle=True)


        self.valid_sampler = DistributedSampler(self.valid_set,shuffle=False) if args.world_size > 1 else None
        self.train_loader = DataLoader(self.train_set,batch_size=args.batch_size,shuffle=False,sampler=self.train_sampler)
        self.valid_loader = DataLoader(self.valid_set,batch_size=args.batch_size,shuffle=False,sampler=self.valid_sampler)

        self.device = args.local_rank if args.world_size > 1 else torch.device('cuda')

        self.model = kprpe_fer(args.kp_rpe_cfg_path,args.token_path,force_download=args.force_download,cos=True)

        if args.ckpt :
            self.model = load_kprpe_finetuned(self.model,args.ckpt)

        self.model = self.model.to(self.device)
        self.aligner = aligners.get_aligner(args.kp_rpe_cfg_path) #if not args.use_hf else get_aligner(args.token_path)
        self.aligner = self.aligner.to(self.device)

        if args.instance_ada_loss or args.cos_margin_loss:
            cos = True if 'RAF' in args.dataset_name else False
            gen_model = kprpe_fer(args.kp_rpe_cfg_path,cos=cos)
            gen_model.load_from_state_dict(args.instance_adaloss_ckpt)
            if args.world_size == 1 or args.rank ==0:
                print('generation model loaded')
            if args.instance_ada_dropout :
                gen_model.train()
            else:
                gen_model.eval()
            self.gen_model = gen_model.to(self.device)



        if args.world_size > 1 :
            self.model = DDP(self.model,device_ids=[args.local_rank],find_unused_parameters=True)
        self.opt = SAM(self.model.parameters(),base_optimizer=torch.optim.AdamW,lr=args.learning_rate, betas=(0.9,0.95),weight_decay=args.learning_rate*90)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt,T_max=args.n_epochs)

        self.best = -1e10
        self.best_macro = -1e10
        self.args = args
        if args.world_size == 1 or args.rank == 0:
            init_wandb(args)
            self.log = []

    def run_train_forward(self,img,label,ldmk):
        _, pred, margin = self.model(img, ldmk)
        bias = 0
        if self.args.instance_ada_loss:
            with torch.no_grad(): # without bias margin and
                _, gen_pred, sims = self.gen_model(img, ldmk)
                j = get_label_noise(label=label, pred=gen_pred) # sparse tensor, (bs, num_classes )
            pred = including_margin(pred,j,0.4)

        if self.args.cos_constant_margin:
            bias+= self.args.cos_constant_margin


        loss = torch.nn.functional.cross_entropy(pred, label) + bias

        if self.args.angle_loss :
            angles = self.model.classifier.get_angles()
            angles = angles.reshape(-1)
            std = torch.std(angles)
            angles = torch.mean(angles)
            loss += self.args.angle_loss*(std*0.7+angles)

        return loss, pred

    def run_train_epoch(self):
        train_acc = 0
        losses = 0
        macro_train_acc = torch.zeros((len(self.valid_set.get_class_counts()))).to(self.device)
        for img, label, q, c_q in tqdm(self.train_loader):
            img = img.to(self.device)
            label = label.to(self.device)
            bs = label.shape[0]
            _, _, ldmk, _, _, _ = self.aligner(img)
            loss,pred = self.run_train_forward(img,label,ldmk)
            loss.backward()
            self.opt.first_step(zero_grad=True)
            with torch.no_grad():
                losses += (bs/len(self.train_set))*loss.item()
                train_acc += (bs / len(self.train_set)) * get_acc(pred, label)
                macro_train_acc += get_macro_acc(pred,label)
            loss,_ = self.run_train_forward(img,label,ldmk)
            loss.backward()
            self.opt.second_step(zero_grad=True)

        if self.args.world_size > 1:
            train_acc = sync(train_acc, self.device)
            losses =sync(losses, self.device)
            macro_train_acc = sync_tensor(macro_train_acc)
        macro_train_acc = torch.mean(macro_train_acc/self.train_set.get_class_counts().to(self.device)).detach().cpu().item()
        return train_acc, macro_train_acc, losses

    @torch.no_grad()
    def run_valid(self):
        acc = 0
        losses = 0
        macro_valid_acc = torch.zeros((len(self.valid_set.get_class_counts()))).to(self.device)
        for img, label, _,_ in tqdm(self.valid_loader):
            img = img.to(self.device)
            label = label.to(self.device)
            _, ldmk, _, _, _, _ = self.aligner(img)
            _, pred, _ = self.model(img, ldmk)
            loss = torch.nn.functional.cross_entropy(pred, label)
            bs = label.shape[0]
            acc += (bs / len(self.valid_set)) * (get_acc(pred, label))
            losses += (bs / len(self.valid_set)) * loss
            macro_valid_acc += get_macro_acc(pred,label)
        if self.args.world_size > 1:
            acc = sync(acc, self.device)
            losses = sync(losses, self.device)
            macro_valid_acc = sync_tensor(macro_valid_acc)

        macro_valid_acc = torch.mean(macro_valid_acc/self.valid_set.get_class_counts().to(self.device)).detach().cpu().item()
        return acc,losses, macro_valid_acc

    def save(self,train_acc,acc,macro_valid_acc,macro_train_acc,train_loss,valid_loss,epoch):
        if self.args.world_size == 1 or self.args.rank == 0:
            wandb.log({'valid_acc': acc, 'train_acc': train_acc, 'macro_valid_acc':macro_valid_acc, 'macro_train_acc':macro_train_acc, 'train_loss':train_loss, 'valid_loss':valid_loss})
            if self.best < acc or self.best_macro < macro_valid_acc or self.args.save_every:
                to_save = self.model.module if self.args.world_size > 1 else self.model
                if self.best < acc :
                    self.best = acc
                    if not self.args.save_every:
                        to_save.kp_rpe.save_pretrained(f'checkpoint/{self.args.name}', name='model.pt')
                        torch.save(to_save.classifier.state_dict(), f'checkpoint/{self.args.name}/classifier.pt')
                    else:
                        save_path = f'checkpoint/{self.args.name}/{epoch}'
                        if not os.path.exists(save_path):
                            os.mkdir(save_path)
                        to_save.kp_rpe.save_pretrained(save_path, name='model.pt')
                        torch.save(to_save.classifier.state_dict(), os.path.join(save_path, 'classifier.pt'))
                elif self.best_macro < macro_valid_acc:
                    self.best_macro = macro_valid_acc
                    to_save.kp_rpe.save_pretrained(f'checkpoint/{self.args.name}', name='macro_model.pt')
                    torch.save(to_save.classifier.state_dict(), f'checkpoint/{self.args.name}/macro_classifier.pt')


    def run_epoch(self,epoch):
        if self.args.world_size > 1:
            self.train_sampler.set_epoch(epoch)
            self.valid_sampler.set_epoch(epoch)
        self.model.train()
        train_acc,macro_train_acc,train_loss = self.run_train_epoch()
        self.scheduler.step()
        self.model.eval()
        valid_acc, valid_loss, macro_valid_acc = self.run_valid()
        self.save(train_acc,valid_acc,macro_valid_acc,macro_train_acc,train_loss,valid_loss,epoch)

    def train(self):
        for epoch in range(self.args.n_epochs):
            self.run_epoch(epoch)
            if self.args.world_size == 1 or self.args.rank == 0 :
                with open(f'checkpoint/{self.args.name}/log.pkl','wb') as f:
                    pickle.dump(self.log,f)

if __name__ == '__main__':
    args = get_args()
    trainer = PreTrainer(args)
    trainer.train()