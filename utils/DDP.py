import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nnF

__all__ = ['sync_scalar', 'concat_all_gather', 'sync_defaultdict', 'gather_tensor']

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nnF  # autograd-enabled collectives

def gather_tensor(feats: torch.Tensor) -> torch.Tensor:
    """
    feats: (N_local, D) with requires_grad possibly True.
    Returns: (N_total, D) where gradients from every chunk are routed back to the owning rank.
    - Uniform local batch: autograd-enabled all_gather then concat.
    - Uneven local batch: pad to max length, gather feats and mask with autograd, then filter by mask.
    """
    # Single-process or not initialized distributed -> passthrough
    if not dist.is_available() or not dist.is_initialized():
        return feats.contiguous()

    world = dist.get_world_size()
    if world == 1:
        return feats.contiguous()

    # 1) Gather local sizes (small scalars, autograd not needed)
    n_local = torch.tensor([feats.shape[0]], device=feats.device, dtype=torch.int64)
    sizes_list = [torch.zeros_like(n_local) for _ in range(world)]
    dist.all_gather(sizes_list, n_local)
    sizes = torch.stack(sizes_list, dim=0).squeeze(-1)  # (world,)

    # 2) Uniform case: same local batch size on all ranks
    if bool(torch.equal(sizes, sizes[0].expand_as(sizes))):
        with torch.autograd.set_detect_anomaly(True):
            gathered = dist_nnF.all_gather(feats)  # tuple of (N_local, D), autograd-enabled
            feats_all = torch.cat(list(gathered), dim=0).contiguous()
            return feats_all

    # 3) Uneven case: pad + mask + autograd-enabled gather
    max_n = int(sizes.max().item())
    D = feats.size(1)
    pad_rows = max_n - feats.size(0)

    if pad_rows > 0:
        pad = torch.zeros((pad_rows, D), device=feats.device, dtype=feats.dtype)
        feats_padded = torch.cat([feats, pad], dim=0)
        mask_local = torch.cat([
            torch.ones(feats.size(0), device=feats.device, dtype=torch.bool),
            torch.zeros(pad_rows, device=feats.device, dtype=torch.bool)
        ], dim=0)
    else:
        feats_padded = feats
        mask_local = torch.ones(max_n, device=feats.device, dtype=torch.bool)

    # Some backends prefer numeric types for collectives; use uint8 for mask and reconvert
    gathered_feats = dist_nnF.all_gather(feats_padded)              # tuple of (max_n, D)
    gathered_mask = dist_nnF.all_gather(mask_local.to(torch.uint8)) # tuple of (max_n,)

    feats_all = torch.cat(list(gathered_feats), dim=0)
    mask_all = torch.cat(list(gathered_mask), dim=0).bool()

    # Keep only valid rows; selection is autograd-safe and does not require mask gradients
    feats_all = feats_all[mask_all].contiguous()
    return feats_all

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