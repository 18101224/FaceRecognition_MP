import torch

def get_grad_norm_x(x,model):
    x = x.detach().clone().requires_grad_(True)
    _,logit = model(x)
    grad = torch.autograd.grad(
        outputs=logit,
        inputs=x,
        grad_outputs=torch.ones_like(logit),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    norm = grad.view(grad.size(0),-1).norm(2,dim=1)
    gp = ((norm-1)**2).mean()
    return gp

def get_grad_norm_L(loss,model,ddp):
    module = model.module if ddp else model
    grad = torch.autograd.grad(
        outputs=loss,
        inputs = module.parameters(),
        create_graph=True,
        retain_graph=True
    )
    norm = torch.sqrt(sum(g.norm()**2 for g in grad))
    return norm
