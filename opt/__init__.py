import torch
from torch.optim import Optimizer
import math

__all__ = ['CAdamW', 'SAM', 'WarmupStepLR', 'WarmupCosineAdamW', 'adjust_learning_rate', 'get_scheduler']

class CAdamW(Optimizer):
    def __init__(self,
                 params,
                 lr,
                 betas=(0.9,0.999),
                 eps=1e-6,
                 weight_decay=0.0,
                 correct_bias=True,
                 no_deprecation_warning=False):
        defaults = {'lr':lr, 'betas':betas, 'eps':eps, 'weight_decay':weight_decay,'correct_bias':correct_bias}
        super().__init__(params,defaults)
        self.init_lr = lr

    @torch.no_grad()
    def step(self,closure=None):
        loss = None
        if closure is not None :
            loss = closure()
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None :
                    continue

                grad = p.grad
                state = self.state[p]
                if 'step' not in state :
                    state['step'] = 0

                if 'exp_avg' not in state :
                    state['exp_avg'] = torch.zeros_like(grad)
                    state['exp_avg_sq'] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state['exp_avg'] , state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] +=1

                if group['weight_decay'] > 0.0 :
                    p.add_(p,alpha=(-group['lr']*group['weight_decay']))

                exp_avg.mul_(beta1).add_(grad,alpha=(1.0-beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad,grad,value=1.0-beta2)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr']
                if group['correct_bias']:
                    bias_correction = 1.0 - beta1**state['step']
                    bias_correction2 = 1.0 - beta2**state['step']
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction

                mask = (exp_avg*grad>0).to(grad.dtype)
                mask = mask*(mask.numel()/(mask.sum()+1))
                norm_grad = (exp_avg*mask)/denom
                p.add_(norm_grad,alpha=-step_size)

            return loss


class SAM(Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

class WarmupStepLR:
    """
    Warm-up + step-decay learning-rate scheduler.

    Args
    ----
    optimizer      : torch.optim.* 인스턴스
    initial_lr     : warm-up 종료 시 도달할 목표 학습률
    warmup_epochs  : warm-up 기간(epoch 수)
    decay_epochs   : 리스트. 각 epoch 넘길 때마다 lr *= decay_factor
    decay_factor   : 학습률 감소 배수(예: 0.1 → 10-배 감소)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        initial_lr: float,
        warmup_epochs: int,
        decay_epochs: list[int],
        decay_factor: float = 0.1,
    ):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = sorted(decay_epochs)
        self.decay_factor = decay_factor

        # 최초 lr=0 으로 시작
        for g in self.optimizer.param_groups:
            g["lr"] = 0.0

    def set_lr(self, lr: float) -> None:
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def get_lr(self, epoch: int) -> float:
        # 1) warm-up (선형 증가)
        if epoch < self.warmup_epochs:
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            # 2) warm-up 이후 base lr
            lr = self.initial_lr
            # 3) step-decay 적용
            for e in self.decay_epochs:
                if epoch >= e:
                    lr *= self.decay_factor
        return lr

    def step(self, epoch: int) -> None:
        self.set_lr(self.get_lr(epoch))

    def state_dict(self) -> dict:
        """Scheduler 상태를 딕셔너리로 반환."""
        return {
            "initial_lr": self.initial_lr,
            "warmup_epochs": self.warmup_epochs,
            "decay_epochs": self.decay_epochs,
            "decay_factor": self.decay_factor,
        }

    def load_state_dict(self, state: dict) -> None:
        """저장해 둔 상태를 복원."""
        self.initial_lr    = state["initial_lr"]
        self.warmup_epochs = state["warmup_epochs"]
        self.decay_epochs  = state["decay_epochs"]
        self.decay_factor  = state["decay_factor"]


class WarmupCosineAdamW(Optimizer):
    """
    AdamW + linear warm-up + cosine decay
    ----------------------------------------------------
    lr_max : warm-up 종료 시점이자 cosine 시작 시점의 학습률
    lr_min : cosine 최종 값
    warmup : warm-up step 수
    total  : 전체 training step 수
    **kwargs : AdamW 의 나머지 하이퍼파라미터( weight_decay 등 )
    """
    def __init__(self, params, *, lr_max, lr_min, warmup, total, **kwargs):
        self.lr_max   = float(lr_max)
        self.lr_min   = float(lr_min)
        self.warmup   = int(warmup)
        self.total    = int(total)
        self.step_num = 0

        # AdamW 인스턴스 내부에 보존
        self.inner_opt = torch.optim.AdamW(params, lr=self.lr_max, **kwargs)

        # Optimizer 부모 클래스 초기화 — state_dict 호환을 위해 필요
        super().__init__(self.inner_opt.param_groups, self.inner_opt.defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """한 step 업데이트 + 스케줄러"""
        self.step_num += 1

        # 1) learning-rate 계산
        if self.step_num < self.warmup:             # linear warm-up
            lr = self.lr_max * self.step_num / self.warmup
        else:                                       # cosine decay
            progress = (self.step_num - self.warmup) / (self.total - self.warmup)
            cosine   = 0.5 * (1 + math.cos(math.pi * progress))
            lr = self.lr_min + (self.lr_max - self.lr_min) * cosine

        # 2) 파라미터 그룹 갱신
        for g in self.inner_opt.param_groups:
            g["lr"] = lr

        # 3) AdamW 실제 업데이트 수행
        loss = self.inner_opt.step(closure)
        return loss

    # ↓ 필요 시 AdamW와 동일한 인터페이스 패스-스루
    def zero_grad(self, set_to_none: bool = False):
        self.inner_opt.zero_grad(set_to_none=set_to_none)


def adjust_learning_rate(optimizer, epoch, scheduler, args):
    if 'cifar' in args.dataset_name :
        from opt.cifar import adjust_learning_rate
        return adjust_learning_rate(optimizer, epoch, scheduler, args)
    elif ('imagenet_lt' in args.dataset_name or 'inat' in args.dataset_name) and args.scheduler == 'cosine':
        lr = args.learning_rate
        warmup_epochs = 5
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs + 1) / (args.n_epochs - warmup_epochs + 1)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr 
    elif ('RAF' in args.dataset_name or 'AffectNet' in args.dataset_name):
        scheduler.step()
        return scheduler.get_last_lr()[0]
    else:
        raise ValueError(f'Scheduler {args.scheduler} is not supported for {args.dataset_name}')    
    
def get_scheduler(args, optimizer):
    if 'cifar' in args.dataset_name :
        from opt.cifar import get_scheduler
        return get_scheduler(args, optimizer)
    elif 'imagenet_lt' in args.dataset_name or 'inat' in args.dataset_name:
        return None
    elif ('RAF' in args.dataset_name or 'AffectNet' in args.dataset_name):
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    else:
        raise ValueError(f'Scheduler {args.scheduler} is not supported for {args.dataset_name}')
    
def get_optimizer(args, model):
    if 'cifar' in args.dataset_name or 'imagenet_lt' in args.dataset_name or 'inat' in args.dataset_name:
        return torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    else:
        return SAM(params=model.parameters(), base_optimizer=torch.optim.AdamW, rho=0.05, adaptive=True, lr=args.learning_rate, weight_decay=args.weight_decay)
    

