import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from dataset import FER, ImbalancedDatasetSampler, DistributedSamplerWrapper, get_multi_view_transforms, CINIC10, CORe50, SmallNORB
from models import ImbalancedModel, dim_dict
from torch import distributed as dist
from argparse import ArgumentParser
from tqdm import tqdm
from utils import get_acc, get_ldmk, sync_scalar, get_macro_acc
from datetime import timedelta
from aligners import get_aligner
import os
import torch.nn.functional as F
from quantization import BRECQOptimizer



def include(loss, losses):
    to_check = loss.split('_')
    for loss_name in losses :
        for loss in to_check :
            if loss_name == loss :
                return True
    return False

def get_model(args):
    if args.ckpt_path is not None or args.resume_path is not None:
        path = args.ckpt_path if args.ckpt_path is not None else args.resume_path
        
        model_params = torch.load(args.ckpt_path, weights_only=False, map_location=torch.device('cpu'))['model_params']
        model = ImbalancedModel(**model_params)
        model.load_from_state_dict(args.ckpt_path, clear_weight=args.clear_classifier)
    else:
        model_params = {
            'num_classes': args.num_classes,
            'model_type': args.model_type,
            'feature_branch': args.feature_branch,
            'feature_module': False, ##UUU
            'regular_simplex': False,
            'cos': True,
            'learnable_input_dist': False,
            'input_layer': False,
            'freeze_backbone': False,
            'remain_backbone': False,
            'decomposition': False,
            'img_size': args.img_size,
            'use_bn': args.use_bn,
        }
        model = ImbalancedModel(**model_params)
    aligner = get_aligner('checkpoint/adaface_vit_base_kprpe_webface12m').cuda() if 'kprpe' in args.model_type else None
    return model.cuda() if args.world_size ==1 else DDP(model.cuda(), device_ids=[args.local_rank], find_unused_parameters=True), \
         aligner, model_params

def get_loaders(args):
    ds = {
        'RAF-DB': FER,
        'AffectNet': FER,
        'CAER': FER,
        'CINIC10': CINIC10,
        'CORe50': CORe50,
        'SmallNORB': SmallNORB
    }[args.dataset_name]
    train_transform, valid_transform, train_transform_wo_aug = get_multi_view_transforms(args, train=True,model_type=args.model_type), \
        get_multi_view_transforms(args, train=False,model_type=args.model_type),\
        get_multi_view_transforms(args, train=False,model_type=args.model_type)
    train_dataset, valid_dataset, train_dataset_wo_aug = ds(args=args, train=True, transform=train_transform, idx=False, imb_factor=args.imb_factor), ds(args=args, train=False, transform=valid_transform, idx=False), ds(args=args, train=False, transform=train_transform_wo_aug, idx=False, balanced=False,imb_factor=1)

    if args.world_size > 1 :
        if args.use_sampler :
            train_sampler = DistributedSamplerWrapper(ImbalancedDatasetSampler(train_dataset, labels=train_dataset.labels), shuffle=True)
        else:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
        valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
        train_sampler_wo_aug = DistributedSampler(train_dataset_wo_aug, shuffle=False)
    else:
        if args.use_sampler :
            train_sampler = ImbalancedDatasetSampler(train_dataset, labels=train_dataset.labels)
        else:
            train_sampler = None
        valid_sampler = None
        train_sampler_wo_aug = None
    train_loader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, shuffle=train_sampler is None, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=128, sampler=valid_sampler, num_workers=args.num_workers, pin_memory=True)
    train_loader_wo_aug = DataLoader(train_dataset_wo_aug, batch_size=128, sampler=train_sampler_wo_aug, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    return train_loader, valid_loader, train_loader_wo_aug


def get_args():
    args = ArgumentParser()

    # distributed setting
    args.add_argument('--world_size', type=int, default=1)
    args.add_argument('--num_workers', type=int, default=0)
    args.add_argument('--use_tf', default=False)
    args.add_argument('--local_rank', type=int, default=None)
    args.add_argument('--rank', type=int, default=None)

    # dataset info
    args.add_argument('--dataset_name', type=str, choices=['RAF-DB', 'AffectNet', 'CAER','CINIC10','CORe50','SmallNORB',], required=True)
    args.add_argument('--dataset_path', type=str, required=True)
    args.add_argument('--num_classes', type=int, default=7)
    args.add_argument('--img_size', type=int, choices=[112,224], default=112)

    # ckpts
    args.add_argument('--resume_path', type=str, default=None)

    # model info
    args.add_argument('--model_type', type=str, choices=['ir50', 'kprpe12m', 'kprpe4m', 'fmae_small', 'Pyramid_ir50','MoCov3','Dinov2'], required=True)
    args.add_argument('--feature_branch', default=False)
    args.add_argument('--ckpt_path', type=str, default=None )
    args.add_argument('--use_bn', default=False)
    args.add_argument('--clear_classifier', default=False, action='store_true')
    args.add_argument('--use_sampler', default=False, action='store_true')
    args.add_argument('--imb_factor', type=float, default=1.0)

    # BRECQ quantization
    args.add_argument('--quantize', action='store_true',
                      help='Run BRECQ post-training quantization on the backbone')
    args.add_argument('--w_bits', type=int, default=4,
                      help='Weight bit-width for body Conv layers')
    args.add_argument('--a_bits', type=int, default=4,
                      help='Activation bit-width')
    args.add_argument('--n_iters', type=int, default=20_000,
                      help='BRECQ iterations per block')
    args.add_argument('--lam', type=float, default=1e-2,
                      help='AdaRound regularization weight')
    args.add_argument('--use_fisher', action='store_true',
                      help='Use Fisher-weighted reconstruction loss')
    args.add_argument('--precompute', action='store_true',
                      help='Precompute activations on CPU (fast but ~76GB RAM for 12K images)')
    args.add_argument('--opt_target', type=str,
                      choices=['both', 'weights', 'activations'],
                      default='both',
                      help='Optimize weight rounding variables, activation scales, or both')
    args.add_argument('--reg_reduction', type=str,
                      choices=['sum', 'mean'],
                      default='sum',
                      help='Reduction for AdaRound regularization across weights')
    args.add_argument('--act_init_mode', type=str,
                      choices=['lsq', 'max', 'percentile'],
                      default='lsq',
                      help='Activation scale initialization rule')
    args.add_argument('--act_init_percentile', type=float, default=0.999,
                      help='Percentile used when activation init mode is percentile')
    args.add_argument('--act_init_samples', type=int, default=64,
                      help='Calibration samples used to seed activation scales')
    args.add_argument('--calib_ratio', type=float, default=1.0,
                      help='Fraction of training set for calibration (0,1]. e.g. 0.08 ≈ 1024/12271')
    args.add_argument('--quant_output', type=str, default=None,
                      help='Path to save quantized model')

    args = args.parse_args()
    vars(args)['dim'] = dim_dict[args.model_type][-1 if args.feature_branch else 0]
    if args.world_size > 1 :
        dist.init_process_group('nccl',world_size=args.world_size,
                           timeout=timedelta(minutes=60))
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.rank = int(os.environ['RANK'])
        args.batch_size = args.batch_size // args.world_size
        torch.cuda.set_device(args.local_rank)


    if args.use_tf:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    return args

class Optimization_Evaluator:
    def __init__(self, args):
        self.args = args
        self.model, self.aligner, self.model_params = get_model(args)
        self.train_loader, self.valid_loader, self.train_loader_wo_aug = get_loaders(args)


    @torch.no_grad()
    def run_valid_forward(self, img, label, ldmk=None):
        logit = self.model(img, keypoint=ldmk, features=False)
        loss = torch.nn.functional.cross_entropy(logit, label)
        return loss, logit

    @torch.no_grad()
    def run_valid_epoch(self):
        self.model.eval()
        total_loss = 0
        total_acc = 0
        total_macro_acc = torch.zeros((self.args.num_classes),device=torch.device('cuda')).float()
        for img, label in tqdm(self.valid_loader, disable=self.args.world_size > 1 and self.args.rank != 0,
         desc=f"validating epoch {self.epoch} latest_acc: {(self.log['valid_acc'][-1] if len(self.log['valid_acc']) > 0 else 0):.4f} best_acc: {self.best_acc:.4f}"):
            img, label = img.cuda(), label.cuda()
            ldmk = get_ldmk(img, self.aligner) if self.aligner is not None else None
            loss, logit = self.run_valid_forward(img, label, ldmk)
            total_loss += loss.detach().item()*label.shape[0]
            total_acc += get_acc(logit, label)*label.shape[0]
            total_macro_acc += get_macro_acc(logit, label)

        if self.args.world_size > 1 :
            total_loss = sync_scalar(total_loss)
            total_acc = sync_scalar(total_acc)
            dist.all_reduce(total_macro_acc, op=dist.ReduceOp.SUM)

        N = len(self.valid_loader.dataset)

        total_macro_acc = (total_macro_acc / torch.tensor(self.valid_loader.dataset.get_img_num_per_cls(), device=torch.device('cuda'), dtype=torch.float32)).float().mean().detach().cpu().item()
        return total_acc / N, total_loss/N, total_macro_acc


# ======================================================================
# BRECQ quantization pipeline
# ======================================================================

@torch.no_grad()
def validate(model, valid_loader, num_classes, brecq_opt=None, desc="Validating"):
    """Validate FP32 or quantized model. Returns (acc, loss, macro_acc).

    If brecq_opt is provided, it replaces the backbone forward path
    with the quantized blocks while keeping the cosine classifier in FP32.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_macro = torch.zeros(num_classes, device="cuda")
    N = 0
    per_cls_counts = torch.tensor(
        valid_loader.dataset.get_img_num_per_cls(), device="cuda", dtype=torch.float32
    )

    for img, label in tqdm(valid_loader, desc=desc):
        img, label = img.cuda(), label.cuda()

        if brecq_opt is not None:
            # quantized backbone -> FP32 cosine head
            emb = brecq_opt.forward(img)               # [B, 256]
            emb = F.normalize(emb, dim=-1, eps=1e-6)
            W   = F.normalize(model.weight, dim=0)     # [256, C]
            logit = emb @ W
        else:
            logit = model(img)

        loss = F.cross_entropy(logit, label)
        total_loss    += loss.item() * label.size(0)
        total_correct += (logit.argmax(1) == label).sum().item()
        total_macro   += get_macro_acc(logit, label)
        N += label.size(0)

    acc      = total_correct / N
    avg_loss = total_loss / N
    macro    = (total_macro / per_cls_counts).mean().item()
    return acc, avg_loss, macro


@torch.no_grad()
def validate_embeddings(embed_fn, classifier_weight, valid_loader, num_classes,
                        desc="Validating"):
    """Validate a backbone embedding path with the shared cosine classifier."""
    total_loss = 0.0
    total_correct = 0
    total_macro = torch.zeros(num_classes, device="cuda")
    N = 0
    per_cls_counts = torch.tensor(
        valid_loader.dataset.get_img_num_per_cls(), device="cuda", dtype=torch.float32
    )

    for img, label in tqdm(valid_loader, desc=desc):
        img, label = img.cuda(), label.cuda()
        emb = embed_fn(img)
        emb = F.normalize(emb, dim=-1, eps=1e-6)
        weight = F.normalize(classifier_weight, dim=0)
        logit = emb @ weight

        loss = F.cross_entropy(logit, label)
        total_loss += loss.item() * label.size(0)
        total_correct += (logit.argmax(1) == label).sum().item()
        total_macro += get_macro_acc(logit, label)
        N += label.size(0)

    acc = total_correct / N
    avg_loss = total_loss / N
    macro = (total_macro / per_cls_counts).mean().item()
    return acc, avg_loss, macro


@torch.no_grad()
def prime_quant_act_scales(brecq_opt, calib_loader, n_samples=64):
    """Initialise LSQ activation scales once, before diagnostic evaluation."""
    parts = []
    collected = 0
    for batch in calib_loader:
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        imgs = imgs.cpu()
        if imgs.shape[-1] != 112 or imgs.shape[-2] != 112:
            imgs = F.interpolate(imgs, size=112)

        need = n_samples - collected
        if need <= 0:
            break
        parts.append(imgs[:need])
        collected += min(need, imgs.shape[0])

        if collected >= n_samples:
            break

    imgs = torch.cat(parts, dim=0)

    x = imgs.to(brecq_opt.device)
    for blk in brecq_opt._quant_blocks:
        blk.init_act_quantizers(x)
        x = blk(x)


@torch.no_grad()
def measure_latency(forward_fn, input_shape, n_warmup=20, n_measure=100,
                    device="cuda"):
    """Measure per-batch inference latency using CUDA events.

    Args:
        forward_fn  : callable that takes a single tensor argument
        input_shape : (B, C, H, W)
        n_warmup    : warm-up iterations (not timed)
        n_measure   : timed iterations

    Returns:
        ms_per_batch   : float, milliseconds per batch
        imgs_per_sec   : float, throughput
    """
    dummy = torch.randn(*input_shape, device=device)

    for _ in range(n_warmup):
        forward_fn(dummy)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_measure):
        forward_fn(dummy)
    end.record()
    torch.cuda.synchronize()

    ms_total     = start.elapsed_time(end)            # milliseconds
    ms_per_batch = ms_total / n_measure
    imgs_per_sec = input_shape[0] / (ms_per_batch / 1000.0)
    return ms_per_batch, imgs_per_sec


def run_brecq_pipeline(args):
    """Full BRECQ pipeline: load model -> FP32 eval -> quantize -> quant eval.

    Checkpoint format (saved by wandb-repro Trainer.save()):
        {'model_state_dict': ImbalancedModel state dict,
         'model_params':     constructor kwargs,
         'best_acc', 'best_macro_acc', 'epoch', 'args', ...}
    """
    # ---- load model ----
    model, _, _ = get_model(args)
    unwrapped = model.module if isinstance(model, DDP) else model

    # ---- data loaders ----
    # Validation loader for evaluation
    ds = {'RAF-DB': FER, 'AffectNet': FER, 'CAER': FER,
          'CINIC10': CINIC10, 'CORe50': CORe50, 'SmallNORB': SmallNORB}[args.dataset_name]
    valid_transform = get_multi_view_transforms(args, train=False, model_type=args.model_type)

    valid_dataset = ds(args=args, train=False, transform=valid_transform, idx=False)
    valid_loader  = DataLoader(valid_dataset, batch_size=128,
                               num_workers=args.num_workers, pin_memory=True)

    # Calibration loader: TRAINING set with no augmentation (for BRECQ)
    calib_dataset = ds(args=args, train=True, transform=valid_transform,
                       idx=False, imb_factor=1.0)
    if args.calib_ratio < 1.0:
        n_calib = max(1, int(len(calib_dataset) * args.calib_ratio))
        calib_dataset = torch.utils.data.Subset(
            calib_dataset,
            torch.randperm(len(calib_dataset))[:n_calib].tolist(),
        )
    calib_loader  = DataLoader(calib_dataset, batch_size=128, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)

    # ---- FP32 validation + latency ----
    print("\n" + "="*60)
    print("FP32 EVALUATION")
    print("="*60)
    fp_acc, fp_loss, fp_macro = validate(
        unwrapped, valid_loader, args.num_classes, desc="[FP32] Validating"
    )
    fp_ms, fp_ips = measure_latency(
        lambda x: unwrapped(x),
        input_shape=(128, 3, args.img_size, args.img_size),
    )
    print(f"  Acc:       {fp_acc:.4f}")
    print(f"  Macro Acc: {fp_macro:.4f}")
    print(f"  Loss:      {fp_loss:.4f}")
    print(f"  Latency:   {fp_ms:.2f} ms/batch  ({fp_ips:.0f} img/s)")

    # ---- BRECQ quantization ----
    print("\n" + "="*60)
    print(f"BRECQ QUANTIZATION  (W{args.w_bits}A{args.a_bits}, "
          f"{args.n_iters} iters/block, "
          f"fisher={'ON' if args.use_fisher else 'OFF'})")
    print("="*60)
    print(f"  Calibration data: training set ({len(calib_dataset)} images, no aug)")
    print(f"  Settings: lam={args.lam:g}  reg={args.reg_reduction}  "
          f"opt={args.opt_target}  act_init={args.act_init_mode}  "
          f"act_init_samples={args.act_init_samples}")

    brecq = BRECQOptimizer(
        unwrapped.backbone,
        w_bits=args.w_bits,
        a_bits=args.a_bits,
        first_last_bits=8,
        n_iters=args.n_iters,
        lam=args.lam,
        batch_size=32,
        use_fisher=args.use_fisher,
        precompute=args.precompute,
        opt_target=args.opt_target,
        reg_reduction=args.reg_reduction,
        act_init_mode=args.act_init_mode,
        act_init_percentile=args.act_init_percentile,
        act_init_samples=args.act_init_samples,
        device="cuda",
        verbose=True,
    )

    print("\n" + "="*60)
    print("PRE-QUANTIZATION DIAGNOSTICS")
    print("="*60)
    folded_acc, folded_loss, folded_macro = validate_embeddings(
        lambda x: brecq._fp_model(x)[0],
        unwrapped.weight,
        valid_loader,
        args.num_classes,
        desc="[Folded FP32] Validating",
    )
    print(f"  Folded FP32 teacher  Acc: {folded_acc:.4f}  Macro: {folded_macro:.4f}  Loss: {folded_loss:.4f}")

    prime_quant_act_scales(brecq, calib_loader, n_samples=args.act_init_samples)
    pre_q_acc, pre_q_loss, pre_q_macro = validate_embeddings(
        brecq.forward,
        unwrapped.weight,
        valid_loader,
        args.num_classes,
        desc=f"[Pre-BRECQ W{args.w_bits}A{args.a_bits}] Validating",
    )
    print(f"  Pre-BRECQ quantized  Acc: {pre_q_acc:.4f}  Macro: {pre_q_macro:.4f}  Loss: {pre_q_loss:.4f}")

    brecq.quantize(calib_loader)

    if args.quant_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.quant_output)), exist_ok=True)
        brecq.save(args.quant_output)

    # ---- Quantized validation + latency ----
    print("\n" + "="*60)
    print(f"W{args.w_bits}A{args.a_bits} QUANTIZED EVALUATION")
    print("="*60)
    q_acc, q_loss, q_macro = validate(
        unwrapped, valid_loader, args.num_classes,
        brecq_opt=brecq, desc=f"[W{args.w_bits}A{args.a_bits}] Validating"
    )
    q_ms, q_ips = measure_latency(
        lambda x: brecq.forward(x),
        input_shape=(128, 3, args.img_size, args.img_size),
    )
    print(f"  Acc:       {q_acc:.4f}")
    print(f"  Macro Acc: {q_macro:.4f}")
    print(f"  Loss:      {q_loss:.4f}")
    print(f"  Latency:   {q_ms:.2f} ms/batch  ({q_ips:.0f} img/s)")

    # ---- comparison ----
    print("\n" + "="*60)
    print("COMPARISON   FP32  vs  W{}A{}".format(args.w_bits, args.a_bits))
    print("="*60)
    print(f"  {'':20s} {'FP32':>10s}   {'Quantized':>10s}   {'Delta':>10s}")
    print(f"  {'Acc':20s} {fp_acc:10.4f}   {q_acc:10.4f}   {q_acc - fp_acc:+10.4f}")
    print(f"  {'Macro Acc':20s} {fp_macro:10.4f}   {q_macro:10.4f}   {q_macro - fp_macro:+10.4f}")
    print(f"  {'CE Loss':20s} {fp_loss:10.4f}   {q_loss:10.4f}   {q_loss - fp_loss:+10.4f}")
    print(f"  {'ms/batch':20s} {fp_ms:10.2f}   {q_ms:10.2f}   {q_ms - fp_ms:+10.2f}")
    print(f"  {'img/s':20s} {fp_ips:10.0f}   {q_ips:10.0f}   {q_ips - fp_ips:+10.0f}")
    print()
    print("  Diagnostic checkpoints:")
    print(f"    original_fp32_acc      = {fp_acc:.4f}")
    print(f"    folded_fp32_acc        = {folded_acc:.4f}")
    print(f"    pre_brecq_quant_acc    = {pre_q_acc:.4f}")
    print(f"    post_brecq_quant_acc   = {q_acc:.4f}")
    print()
    print("  NOTE: Latency comparison reflects simulated quantization (FP32 arithmetic")
    print("  with fake-quant ops). Real INT4 speedup requires TensorRT / custom kernels.")


if __name__ == '__main__':
    args = get_args()

    if args.quantize:
        run_brecq_pipeline(args)
    else:
        trainer = Optimization_Evaluator(args)
        trainer.run_valid_epoch()
