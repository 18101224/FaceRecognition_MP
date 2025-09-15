import torch
import torch.nn.functional as F
import torch.distributed as dist

__all__ = ['analyze_and_update_gradients']

def analyze_and_update_gradients(
    model, ce_loss, fr_loss, optimizer, include_backbone=False, world_size=1
):
    """
    DDP 환경과 단일 GPU 환경을 모두 지원하며, 두 손실의 그래디언트 관계를 분석하고
    최종 파라미터 업데이트를 준비합니다.

    - ce_loss: 모델의 모든 파라미터에 그래디언트를 계산합니다.
    - fr_loss: `include_backbone=False`일 경우 'decomposition' 파라미터에만,
               `include_backbone=True`일 경우 'decomposition'과 'backbone.body3'에 그래디언트를 계산합니다.

    Args:
        model (torch.nn.Module): 분석 및 업데이트 대상 모델.
        ce_loss (torch.Tensor): 주 손실 (Cross-Entropy 등).
        fr_loss (torch.Tensor): 보조 손실 (Face Recognition 등).
        optimizer (torch.optim.Optimizer): 모델의 최적화 도구.
        include_backbone (bool): fr_loss와 그래디언트 분석에 'backbone.body3'를 포함할지 여부.
        world_size (int): DDP 환경의 총 프로세스(GPU) 수. 단일 GPU 환경일 경우 1.

    Returns:
        tuple: (cos_sims, norms_ce, norms_fr)
          - cos_sims (dict): 각 모듈을 키로 가지는 (동기화된) 코사인 유사도 값.
          - norms_ce (dict): 각 모듈을 키로 가지는 (동기화된) ce_loss 그래디언트의 L2 놈 값.
          - norms_fr (dict): 각 모듈을 키로 가지는 (동기화된) fr_loss 그래디언트의 L2 놈 값.
    """
    # --- 1. ce_loss 그래디언트 계산 및 저장 ---
    optimizer.zero_grad()
    ce_loss.backward(retain_graph=True)
    grad_storage_ce = {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters() if param.grad is not None
    }

    # --- 2. fr_loss 그래디언트 계산 및 저장 ---
    optimizer.zero_grad()
    params_for_fr = list(model.decomposition.parameters())
    if include_backbone:
        params_for_fr.extend(list(model.backbone.body3.parameters()))
    
    fr_loss.backward(inputs=params_for_fr)
    grad_storage_fr = {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters() if param.grad is not None
    }

    # --- 3. 로컬 그래디언트 분석 ---
    cos_sims, norms_ce, norms_fr = {}, {}, {}
    
    module_prefixes = ['decomposition']
    if include_backbone:
        module_prefixes.append('backbone.body3')

    for prefix in module_prefixes:
        # 각 모듈의 그래디언트를 하나의 벡터로 통합
        ce_vec = torch.cat([g.view(-1) for n, g in grad_storage_ce.items() if n.startswith(prefix)])
        
        # fr_loss는 부분적으로만 그래디언트를 가지므로, 해당 모듈의 그래디언트가 있는지 확인
        if any(n.startswith(prefix) for n in grad_storage_fr):
            fr_vec = torch.cat([g.view(-1) for n, g in grad_storage_fr.items() if n.startswith(prefix)])
        else:
            fr_vec = torch.tensor([], device=ce_vec.device)

        if ce_vec.numel() > 0 and fr_vec.numel() > 0:
            cos_sims[prefix] = F.cosine_similarity(ce_vec, fr_vec, dim=0, eps=1e-8)
            norms_ce[prefix] = ce_vec.norm()
            norms_fr[prefix] = fr_vec.norm()
        else:
            # 분석 불가능한 경우 0으로 채워진 텐서 생성
            cos_sims[prefix] = torch.tensor(0.0, device=next(model.parameters()).device)
            norms_ce[prefix] = torch.tensor(0.0, device=next(model.parameters()).device)
            norms_fr[prefix] = torch.tensor(0.0, device=next(model.parameters()).device)


    # --- 4. DDP 환경일 경우, 분석 값 동기화 ---
    if world_size > 1:
        for prefix in module_prefixes:
            # 동기화할 값들을 하나의 텐서로 묶음
            metrics_to_sync = torch.stack([cos_sims[prefix], norms_ce[prefix], norms_fr[prefix]])
            
            # 모든 GPU의 텐서를 합산
            dist.all_reduce(metrics_to_sync, op=dist.ReduceOp.SUM)
            
            # GPU 수로 나누어 평균 계산
            metrics_synced = metrics_to_sync / world_size
            
            # 동기화된 값을 다시 딕셔너리에 할당
            cos_sims[prefix] = metrics_synced[0].item()
            norms_ce[prefix] = metrics_synced[1].item()
            norms_fr[prefix] = metrics_synced[2].item()
    else:
        # 단일 GPU 환경일 경우, .item()을 이용해 스칼라 값으로 변환
        for prefix in module_prefixes:
            cos_sims[prefix] = cos_sims[prefix].item()
            norms_ce[prefix] = norms_ce[prefix].item()
            norms_fr[prefix] = norms_fr[prefix].item()

    # --- 5. 최종 업데이트를 위해 그래디언트 버퍼 재구성 ---
    optimizer.zero_grad()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            grad_c = grad_storage_ce.get(name)
            grad_f = grad_storage_fr.get(name)
            
            if grad_c is not None or grad_f is not None:
                final_grad = torch.zeros_like(param.data)
                if grad_c is not None:
                    final_grad += grad_c
                if grad_f is not None:
                    final_grad += grad_f
                param.grad = final_grad
    
    # optimizer.step()은 이 함수를 호출한 외부에서 실행
    return cos_sims, norms_ce, norms_fr

