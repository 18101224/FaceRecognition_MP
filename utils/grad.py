import torch
import torch.distributed as dist
from collections import defaultdict
__all__ = ['measure_grad']

def measure_grad(model, l1, l2, log:defaultdict,layer_names=None, eps=1e-12, ddp_reduce='sum'):
    """
    전제:
      - (l1 + l2).backward(retain_graph=True) 호출되어 그래프 유지됨
      - .grad를 읽거나 수정하지 않음 (record-only)
    ddp_reduce: 'sum' | 'mean'  (동기화 스케일 선택; cosine 결과는 동일)
    """
    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if layer_names is not None:
        ln_list = list(layer_names)
        sel = []
        for n, p in named_params:
            if any(n.startswith(ln) or (ln in n) for ln in ln_list):
                sel.append((n, p))
    else:
        sel = named_params
    if not sel:
        return {}

    params = [p for _, p in sel]

    # 손실별 파라미터 그래디언트 계산 (.grad에 누적되지 않음)
    grads_l1 = torch.autograd.grad(
        l1, params, retain_graph=True, allow_unused=True, create_graph=False
    )
    grads_l2 = torch.autograd.grad(
        l2, params, retain_graph=True, allow_unused=True, create_graph=False
    )

    def key_of(n):
        if layer_names is None:
            return n
        for ln in layer_names:
            if n.startswith(ln) or (ln in n):
                return ln
        return None

    device = params[0].device
    agg = {}
    for (name, p), g1, g2 in zip(sel, grads_l1, grads_l2):
        k = key_of(name)
        if k is None:
            continue
        if g1 is None:
            g1 = torch.zeros_like(p, device=device)
        if g2 is None:
            g2 = torch.zeros_like(p, device=device)
        g1v = g1.detach().reshape(-1)
        g2v = g2.detach().reshape(-1)
        if k not in agg:
            agg[k] = {
                'dot': torch.zeros((), device=device),
                's1': torch.zeros((), device=device),
                's2': torch.zeros((), device=device),
            }
        agg[k]['dot'] += (g1v * g2v).sum()
        agg[k]['s1'] += (g1v * g1v).sum()
        agg[k]['s2'] += (g2v * g2v).sum()

    # DDP 동기화 (in-place all_reduce)
    is_ddp = dist.is_available() and dist.is_initialized()
    if is_ddp:
        world_size = dist.get_world_size()
        for v in agg.values():
            dist.all_reduce(v['dot'], op=dist.ReduceOp.SUM)
            dist.all_reduce(v['s1'],  op=dist.ReduceOp.SUM)
            dist.all_reduce(v['s2'],  op=dist.ReduceOp.SUM)
            if ddp_reduce == 'mean' and world_size > 1:
                inv = 1.0 / world_size
                v['dot'] *= inv
                v['s1']  *= inv
                v['s2']  *= inv

    for k, v in agg.items():
        n1 = torch.sqrt(v['s1']).item()
        n2 = torch.sqrt(v['s2']).item()
        denom = max(n1 * n2, eps)
        cos = (v['dot'].item() / denom) if denom > 0.0 else 0.0
        # logging: save per-layer dot and norms (and raw sumsq) into defaultdict log
        if log is not None:
            key_dot = f'{k}_dot'
            key_l1_norm = f'{k}_l1_norm'
            key_l2_norm = f'{k}_l2_norm'
            key_l1_sumsq = f'{k}_l1_sumsq'
            key_l2_sumsq = f'{k}_l2_sumsq'
            log[key_dot].append(v['dot'].item())
            log[key_l1_norm].append(n1)
            log[key_l2_norm].append(n2)
            log[key_l1_sumsq].append(v['s1'].item())
            log[key_l2_sumsq].append(v['s2'].item())

    return log 
