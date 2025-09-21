import torch
import torch.distributed as dist

__all__ = ['sync_scalar', 'concat_all_gather', 'sync_defaultdict']


def sync_defaultdict(dictionary, N,normalize: bool = False):
    for key, value in dictionary.items():
        dictionary[key] = sync_scalar(value, normalize=normalize)/N
    return dictionary


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

def concat_all_gather(tensor):
    """
    All-gather 1st-dimension variable-length tensors across DDP ranks.

    - If DDP is not initialized, returns the input tensor.
    - Pads each local tensor along dim=0 to the global max length,
      gathers, trims the padding per rank, and finally concatenates.
    - Works on CPU or GPU depending on the input tensor device.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return tensor

    world_size = dist.get_world_size()

    # Ensure at least 1 dimension
    if tensor.dim() == 0:
        tensor = tensor.unsqueeze(0)

    device = tensor.device
    dtype = tensor.dtype

    # Gather local sizes
    local_size = torch.tensor([tensor.size(0)], device=device, dtype=torch.int64)
    size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)

    sizes = torch.stack(size_list, dim=0)  # [world_size, 1]
    max_size = int(sizes.max().item())

    # Pad along dim=0 to max_size
    if tensor.size(0) < max_size:
        pad_shape = (max_size - tensor.size(0),) + tuple(tensor.shape[1:])
        padding = torch.zeros(pad_shape, device=device, dtype=dtype)
        padded = torch.cat([tensor, padding], dim=0)
    else:
        padded = tensor

    # All-gather padded tensors
    gather_list = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gather_list, padded)

    # Trim padding per rank and concatenate
    trimmed = []
    for rank_idx in range(world_size):
        valid = int(sizes[rank_idx].item())
        if valid > 0:
            trimmed.append(gather_list[rank_idx][:valid])
    if len(trimmed) == 0:
        # All ranks had zero length
        return tensor.new_empty((0,) + tuple(tensor.shape[1:]))

    return torch.cat(trimmed, dim=0)