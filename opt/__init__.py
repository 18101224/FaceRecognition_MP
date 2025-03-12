import torch
from torch.optim import Optimizer
import math

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

