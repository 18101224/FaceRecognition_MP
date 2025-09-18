# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F



class KCL:
    """
    K-positive Contrastive Loss (멀티-포지티브 InfoNCE/SupCon 일반화)
    
    Input Shapes:
    - features: [N, D] - 앵커 임베딩 (L2 정규화 권장)
    - labels: [N] - 앵커 라벨 (0..C-1)
    - pos_feats: [N, K, D] 또는 List of [k_i, D] - K개(또는 가변) 양성 임베딩
    - pos_labels: [N, K] 또는 List of [k_i] - 양성 라벨
    
    Return Shape:
    - loss: scalar tensor - 배치 평균(또는 합) 손실값
    
    Arguments:
    - exclude_same_class_from_negatives: True이면 배치 내 같은 클래스를 negative에서 제외
    - include_positives_in_denominator: True이면 외부 positive를 분모에 포함 (원래 SupCon 방식)
    """
    
    def __init__(self,  
                 temperature: float = 0.07,
                 include_positives_in_denominator: bool = False,    # 외부 positive를 분모에 포함할지
                 exclude_same_class_from_negatives: bool = True,    # 배치 내 같은 클래스를 negative에서 제외할지
                 use_batch_negatives: bool = True,
                 reduction: str = "mean"):
        """
        Initialize K-positive Contrastive Loss (KCL).

        Arguments
        - temperature (float): Softmax temperature τ used to scale all
          similarity scores. Smaller values sharpen the distribution.
        - include_positives_in_denominator (bool): When True, all positive
          similarities used in the numerator are also included in the
          denominator (SupCon-style). When False, the denominator uses only
          negatives (and optionally batch negatives depending on the flags).
        - exclude_same_class_from_negatives (bool): If True, samples that share
          the same label as the anchor are excluded from the negative set when
          building the denominator. If False, all non-self samples can act as
          negatives.
        - use_batch_negatives (bool): If True, additionally use in-batch
          negatives (anchors from other samples in the batch) in the
          denominator. If False, only external negatives (if any) are used.
        - reduction (str): Aggregation over the batch. 'mean' to average the
          per-sample losses, 'sum' to sum them.

        Attributes
        - tau (float): Stored temperature value.
        - include_pos_in_den (bool): Whether to include positives in the denom.
        - exclude_same_class (bool): Whether same-label pairs are excluded from
          negatives.
        - use_batch_negs (bool): Whether to use batch negatives.
        - reduction (str): Reduction to apply to the final loss.
        """
        self.tau = float(temperature)                              # InfoNCE 온도 파라미터
        self.include_pos_in_den = bool(include_positives_in_denominator)  # 외부 양성을 분모에 포함
        self.exclude_same_class = bool(exclude_same_class_from_negatives)  # 같은 클래스를 음성에서 제외
        self.use_batch_negs = bool(use_batch_negatives)           # 배치 음성 사용 여부
        assert reduction in ("mean", "sum")
        self.reduction = reduction                                 # 손실 축약 방식

    def __call__(self,
                 features: torch.Tensor,      # [N, D] 앵커 임베딩
                 labels: torch.Tensor,        # [N] 앵커 라벨
                 pos_feats,                   # [N, K, D] 또는 List of [k_i, D] 양성 임베딩
                 pos_labels                   # [N, K] 또는 List of [k_i] 양성 라벨
                 ) -> torch.Tensor:           # scalar 손실값
        
        N, D = features.shape                                      # N: 배치 크기, D: 임베딩 차원
  # [N, D] 정규화된 앵커
        
        # 입력 형태 판별: 고정 길이 vs 가변 길이
        if isinstance(pos_feats, torch.Tensor) and pos_feats.dim() == 3:
            # 고정 길이 [N, K, D] 케이스
            return self.compute_fixed(features, pos_feats, labels)
        elif isinstance(pos_feats, list):
            # 가변 길이 List of [k_i, D] 케이스
            return self._compute_variable_length(q, labels, pos_feats, pos_labels)
        else:
            raise ValueError("pos_feats must be [N, K, D] tensor or list of [k_i, D] tensors")

    def _compute_fixed_length(self, q, labels, pos_feats, pos_labels):
        """
        고정 길이 [N, K, D] 양성 처리 (배치 내 같은 클래스 고려)
        
        Args:
            q: [N, D] 정규화된 앵커 임베딩
            labels: [N] 앵커 라벨
            pos_feats: [N, K, D] 외부 양성 임베딩 (MoCo 큐에서)
            pos_labels: [N, K] 외부 양성 라벨
        
        Returns:
            loss: scalar 손실값
        """
        N = q.size(0)                                              # N: 배치 크기
        K = pos_feats.size(1)                                      # K: 앵커당 외부 양성 수
        P = pos_feats  # [N, K, D] 정규화된 외부 양성
        
        # 외부 양성 유사도
        s_pos_external = torch.einsum("nd,nkd->nk", q, P) / self.tau    # [N, K] 앵커-외부양성 유사도
        
        # 배치 내 같은 클래스 관계 파악
        labels_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)       # [N, N] 같은 클래스 마스크
        eye = torch.eye(N, device=q.device, dtype=torch.bool)           # [N, N] 자기 자신 마스크
        same_class_mask = labels_matrix & (~eye)                         # [N, N] 같은 클래스 (자기 자신 제외)
        
        # 배치 내 모든 쌍 유사도
        s_batch = torch.mm(q, q.t()) / self.tau                         # [N, N] 모든 배치 유사도
        
        # 각 앵커별로 개별 처리 (같은 클래스 수가 다를 수 있음)
        losses = []                                                      # 각 앵커의 개별 손실
        
        for i in range(N):
            # === 분자: 모든 양성 (외부 + 배치 내 같은 클래스) ===
            pos_parts = [s_pos_external[i]]                             # [K] 외부 양성
            
            # 배치 내 같은 클래스 양성 추가
            batch_same_class = same_class_mask[i]                        # [N] i번째 앵커와 같은 클래스 마스크
            if batch_same_class.any():
                batch_pos = s_batch[i][batch_same_class]                 # [M_i] 배치 내 같은 클래스 유사도
                pos_parts.append(batch_pos)
            
            all_positives = torch.cat(pos_parts, dim=0)                  # [K + M_i] 모든 양성
            num_i = torch.logsumexp(all_positives, dim=0)                # scalar 분자 log-sum-exp
            
            # === 분모: 양성(선택) + 음성 ===
            den_parts = []                                               # 분모 구성 요소들
            
            # 1) 외부 양성을 분모에 포함할지 (원래 SupCon 방식)
            if self.include_pos_in_den:
                den_parts.append(all_positives)                         # [K + M_i] 모든 양성
            
            # 2) 배치 음성 처리
            if self.use_batch_negs:
                if self.exclude_same_class:
                    # 같은 클래스를 음성에서 제외 (더 합리적)
                    diff_class_mask = ~labels_matrix[i] & ~eye[i]        # [N] 다른 클래스 (자기 자신 제외)
                    if diff_class_mask.any():
                        batch_neg = s_batch[i][diff_class_mask]          # [L_i] 다른 클래스만 음성
                        den_parts.append(batch_neg)
                else:
                    # 모든 배치 샘플을 음성으로 (자기 자신만 제외, 원래 방식)
                    batch_neg = s_batch[i][~eye[i]]                      # [N-1] 자기 자신 제외한 모든 샘플
                    den_parts.append(batch_neg)
            
            # 분모가 비어있으면 최소한 양성은 포함 (정의역 보장)
            if len(den_parts) == 0 or all(len(part) == 0 for part in den_parts):
                den_parts = [all_positives]                              # [K + M_i] 양성으로 폴백
                
            den_concat = torch.cat(den_parts, dim=0)                     # [varies] 분모 후보들
            den_i = torch.logsumexp(den_concat, dim=0)                   # scalar 분모 log-sum-exp
            
            # i번째 앵커의 손실
            losses.append(-(num_i - den_i))                              # scalar 음의 로그 확률
        
        loss_tensor = torch.stack(losses)                               # [N] 모든 앵커의 손실
        return loss_tensor.mean() if self.reduction == "mean" else loss_tensor.sum()  # scalar 최종 손실

    def _compute_variable_length(self, q, labels, pos_feats_list, pos_labels_list):
        """
        가변 길이 List of [k_i, D] 양성 처리 (배치 내 같은 클래스 고려)
        
        Args:
            q: [N, D] 정규화된 앵커 임베딩
            labels: [N] 앵커 라벨
            pos_feats_list: List[Tensor] 길이 N, 각 [k_i, D] 외부 양성
            pos_labels_list: List[Tensor] 길이 N, 각 [k_i] 외부 양성 라벨
        
        Returns:
            loss: scalar 손실값
        """
        N = q.size(0)                                                    # N: 배치 크기
        assert len(pos_feats_list) == N, "pos_feats_list length must match N"
        
        # 배치 내 같은 클래스 관계 파악
        labels_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)       # [N, N] 같은 클래스 마스크
        eye = torch.eye(N, device=q.device, dtype=torch.bool)           # [N, N] 자기 자신 마스크
        same_class_mask = labels_matrix & (~eye)                         # [N, N] 같은 클래스 (자기 자신 제외)
        
        # 배치 내 모든 쌍 유사도
        s_batch = torch.mm(q, q.t()) / self.tau                         # [N, N] 모든 배치 유사도
        
        losses = []                                                      # 각 앵커의 개별 손실
        
        for i in range(N):
            # === 외부 양성 처리 ===
            P_i = pos_feats_list[i]                                     # [k_i, D] i번째 앵커의 외부 양성
            if P_i.numel() == 0:
                # 외부 양성이 없으면 0 손실
                losses.append(torch.tensor(0.0, device=q.device))       # scalar 0 손실
                continue
                
            P_i = F.normalize(P_i, dim=1) if self.normalize else P_i     # [k_i, D] 정규화된 외부 양성
            s_pos_external = torch.mv(P_i, q[i]) / self.tau             # [k_i] 앵커-외부양성 유사도
            
            # === 분자: 모든 양성 (외부 + 배치 내 같은 클래스) ===
            pos_parts = [s_pos_external]                                # [k_i] 외부 양성
            
            # 배치 내 같은 클래스 양성 추가
            batch_same_class = same_class_mask[i]                        # [N] i번째 앵커와 같은 클래스 마스크
            if batch_same_class.any():
                batch_pos = s_batch[i][batch_same_class]                 # [M_i] 배치 내 같은 클래스 유사도
                pos_parts.append(batch_pos)
            
            all_positives = torch.cat(pos_parts, dim=0)                  # [k_i + M_i] 모든 양성
            num_i = torch.logsumexp(all_positives, dim=0)                # scalar 분자 log-sum-exp
            
            # === 분모: 양성(선택) + 음성 ===
            den_parts = []                                               # 분모 구성 요소들
            
            # 1) 외부 양성을 분모에 포함할지
            if self.include_pos_in_den:
                den_parts.append(all_positives)                         # [k_i + M_i] 모든 양성
            
            # 2) 배치 음성 처리
            if self.use_batch_negs:
                if self.exclude_same_class:
                    # 같은 클래스를 음성에서 제외
                    diff_class_mask = ~labels_matrix[i] & ~eye[i]        # [N] 다른 클래스 (자기 자신 제외)
                    if diff_class_mask.any():
                        batch_neg = s_batch[i][diff_class_mask]          # [L_i] 다른 클래스만 음성
                        den_parts.append(batch_neg)
                else:
                    # 모든 배치 샘플을 음성으로 (자기 자신만 제외)
                    batch_neg = s_batch[i][~eye[i]]                      # [N-1] 자기 자신 제외한 모든 샘플
                    den_parts.append(batch_neg)
            
            # 분모가 비어있으면 양성으로 폴백
            if len(den_parts) == 0 or all(len(part) == 0 for part in den_parts):
                den_parts = [all_positives]                              # [k_i + M_i] 양성으로 폴백
                
            den_concat = torch.cat(den_parts, dim=0)                     # [varies] 분모 후보들
            den_i = torch.logsumexp(den_concat, dim=0)                   # scalar 분모 log-sum-exp
            
            # i번째 앵커의 손실
            losses.append(-(num_i - den_i))                             # scalar 음의 로그 확률
        
        loss_tensor = torch.stack(losses)                               # [N] 모든 앵커의 손실
        return loss_tensor.mean() if self.reduction == "mean" else loss_tensor.sum()  # scalar 최종 손실



    def compute_fixed(self,q,k,y):
        N,K,D = k.shape 
        k_sims = (q.unsqueeze(1) @ k.transpose(-1,-2)).squeeze(1) / self.tau  # N, k 

        batch_sims = q@q.T / self.tau # N, N 

        positive_batchs = y.unsqueeze(1) == y.unsqueeze(0)
        neg_batches = ~positive_batchs  # N,N 
        eye = torch.eye(N, device=q.device, dtype=torch.bool)
        same_class_mask = positive_batchs & (~eye) # N,N 
        same_class_counts = same_class_mask.sum(dim=1).reshape(N) + K # N 

        batch_pos_sims = batch_sims * same_class_mask
        batch_pos_sims = batch_pos_sims.masked_fill(batch_pos_sims == 0, float('-inf'))  # N, N 

        pos_sims = torch.cat([k_sims, batch_pos_sims],dim=1 )
        y_counts = y.bincount().to(q.device) # C 
        neg_sims = batch_sims*neg_batches # N,N * N,N -> N,N 
        neg_sims = neg_sims.masked_fill(neg_sims == 0, float('-inf'))  # N, N 

        num = torch.logsumexp(pos_sims, dim=1) # N 
        den = torch.log(torch.sum(torch.exp(neg_sims) / y_counts[y].reshape(1,-1),dim=1).reshape(N,))

        loss = -(num - den) / same_class_counts 

        return loss.mean()

