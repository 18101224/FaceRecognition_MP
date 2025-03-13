from dataset import FER
from models import kprpe_fer, load_kprpe_finetuned
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import get_acc,sync, get_label_noise
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
from Loss import *

def get_args():
    args = ArgumentParser()
    args.add_argument('--LDAM',default=False,help='LDAM')
    args.add_argument('--cos_margin_loss',default=False,help='class angular margin')
    args.add_argument('--class_quality_loss',default=False,help='class quality loss')
    args.add_argument('--class_ada_loss',default=False,help='class proportion')
    args.add_argument('--ldam_weight',type=float)
    args.add_argument('--proportion_alpha',type=float)
    args.add_argument('--quality_beta',type=float)
    args.add_argument('--margin',type=float)
    args.add_argument('--instance_ada_loss',default=False)
    args.add_argument('--instance_adaloss_ckpt')
    args.add_argument('--instance_ada_dropout',default=False)
    args.add_argument('--instance_alpha',type=float)
    args.add_argument('--class_alpha',type=float)
    args.add_argument('--learning_rate',type=float)
    args.add_argument('--batch_size',type=int)
    args.add_argument('--world_size',default=1,type=int)
    args.add_argument('--use_hf',default=False)
    args.add_argument('--token_path',default=None)
    args.add_argument('--kp_rpe_cfg_path')
    args.add_argument('--dataset_name')
    args.add_argument('--quality_model_path')
    args.add_argument('--name',default=False)
    args.add_argument('--dataset_path')
    args.add_argument('--wandb_token')
    args.add_argument('--local_rank')
    args.add_argument('--ckpt')
    args.add_argument('--force_download',default=False)
    args.add_argument('--rank')
    args.add_argument('--save_every',default=False)
    args.add_argument('--n_epochs', type=int)
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
        self.train_set = raf_for_kp(args.dataset_path,args.quality_model_path) if 'RAF' in args.dataset_path else FER(args,args.quality_model_path)
        self.valid_set = raf_for_kp(args.dataset_path,args.quality_model_path, train=False) if 'RAF' in args.dataset_path else  FER(args,args.quality_model_path,train=False)
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
            gen_model = kprpe_fer(args.kp_rpe_cfg_path,cos=False)
            gen_model.load_from_state_dict(args.instance_adaloss_ckpt)
            if args.world_size == 1 or args.rank ==0:
                print('generation model loaded')
            if args.instance_ada_dropout :
                gen_model.train()
            else:
                gen_model.eval()
            self.gen_model = gen_model.to(self.device)


        self.class_ada_loss = Proportion_loss(self.train_set.labels,args.proportion_alpha,self.device)

        if args.world_size > 1 :
            self.model = DDP(self.model,device_ids=[args.local_rank],find_unused_parameters=True)
        self.opt = SAM(self.model.parameters(),base_optimizer=torch.optim.AdamW,lr=args.learning_rate, betas=(0.9,0.95),weight_decay=args.learning_rate*90)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt,milestones=[i for i in range(args.n_epochs//3,args.n_epochs,args.n_epochs//3)],gamma=0.1)

        self.best = -1e10
        self.args = args
        if args.world_size == 1 or args.rank == 0:
            init_wandb(args)
            self.log = []

    def run_train_forward(self,img,label,ldmk):
        _, pred, margin = self.model(img, ldmk)
        coef = 1
        with torch.no_grad():
            if self.args.class_ada_loss:
                a = self.class_ada_loss.get_proportion_weights(label)
                coef = coef * a
            if self.args.instance_ada_loss:
                _, gen_pred = self.gen_model(img, ldmk)
                a = compute_instance_adaloss(gen_pred, pred, alpha=self.args.instance_alpha)
                coef = coef * a
            #if self.args.class_quality_loss:
            #    a = 1 - self.args.quality_beta * c_q.to(self.device)
            #    coef = coef * a
            if self.args.cos_margin_loss:
                if not self.args.instance_ada_loss:
                    _, gen_pred = self.gen_model(img, ldmk)
                j = get_label_noise(gen_pred, label)
                gamma = self.class_ada_loss.get_gamma(label)
                angle = margin[label]

                cos = margin_logit(cos=pred[torch.arange(label.shape[0]),label],
                                    j=j,
                                    angle=angle,
                                    m=self.args.margin,
                                    gamma=gamma)
                pred[torch.arange(label.shape[0]),label] = cos.to(pred.dtype)
        pred = torch.nn.functional.softmax(pred,dim=-1)
        loss = torch.nn.functional.cross_entropy(pred, label, reduction='none') * coef
        loss = torch.mean(loss, dim=0, keepdim=False)
        return loss, pred

    def run_train_epoch(self):
        train_acc = 0
        for img, label, q, c_q in tqdm(self.train_loader):
            img = img.to(self.device)
            label = label.to(self.device)
            bs = label.shape[0]
            _, _, ldmk, _, _, _ = self.aligner(img)
            loss,pred = self.run_train_forward(img,label,ldmk)
            loss.backward()
            self.opt.first_step(zero_grad=True)
            with torch.no_grad():
                train_acc += (bs / len(self.train_set)) * get_acc(pred, label)
            loss,_ = self.run_train_forward(img,label,ldmk)
            loss.backward()
            self.opt.second_step(zero_grad=True)

        if self.args.world_size > 1:
            train_acc = sync(train_acc, self.device)

        return train_acc

    @torch.no_grad()
    def run_valid(self):
        acc = 0
        losses = 0
        for img, label, _,_ in tqdm(self.valid_loader):
            img = img.to(self.device)
            label = label.to(self.device)
            _, ldmk, _, _, _, _ = self.aligner(img)
            _, pred, _ = self.model(img, ldmk)
            loss = torch.nn.functional.cross_entropy(pred, label)
            bs = label.shape[0]
            acc += (bs / len(self.valid_set)) * (get_acc(pred, label))
            losses += (bs / len(self.valid_set)) * loss
        if self.args.world_size > 1:
            acc = sync(acc, self.device)
            losses = sync(losses, self.device)
        return acc,losses

    def save(self,train_acc,acc,epoch):
        if self.args.world_size == 1 or self.args.rank == 0:
            wandb.log({'valid_acc': acc, 'train_acc': train_acc})
            if self.best < acc or self.args.save_every:
                self.best = acc
                to_save = self.model.module if self.args.world_size > 1 else self.model
                if not self.args.save_every:
                    to_save.kp_rpe.save_pretrained(f'checkpoint/{self.args.name}', name='model.pt')
                    torch.save(to_save.classifier.state_dict(), f'checkpoint/{self.args.name}/classifier.pt')
                else:
                    save_path = f'checkpoint/{self.args.name}/{epoch}'
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                    to_save.kp_rpe.save_pretrained(save_path, name='model.pt')
                    torch.save(to_save.classifier.state_dict(), os.path.join(save_path, 'classifier.pt'))

    def run_epoch(self,epoch):
        if self.args.world_size > 1:
            self.train_sampler.set_epoch(epoch)
            self.valid_sampler.set_epoch(epoch)
        self.model.train()
        train_acc = self.run_train_epoch()
        self.model.eval()
        valid_acc, valid_loss = self.run_valid()
        self.save(train_acc,valid_acc,epoch)

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