import torch
import torch.distributed as dist

__all__ = ['sync_scalar']

def sync_scalar(scalar, normalize: bool = False ):
    """
    All-reduce a scalar across ranks and optionally average.

    - Accepts Python float/int or a torch scalar (0-dim or 1-element tensor).
    - If torch.distributed is not initialized, returns the input as float.
    - If normalize is True, divides by world_size after summation.
    """
    # Fast path: no dist initialized
    if not (dist.is_available() and dist.is_initialized()):
        if isinstance(scalar, torch.Tensor):
            return float(scalar.detach().cpu().item())
        return float(scalar)

    # Ensure tensor on the right device/dtype
    if isinstance(scalar, torch.Tensor):
        tensor = scalar.detach().to(dtype=torch.float32)
        if tensor.dim() == 0:
            tensor = tensor.clone()
        else:
            tensor = tensor.view(1).clone()
    else:
        tensor = torch.tensor(float(scalar), dtype=torch.float32)

    tensor = tensor.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # All-reduce (sum)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    if normalize:
        world_size = dist.get_world_size()
        tensor /= world_size

    return float(tensor.cpu().item())