import torch
import torch.nn.functional as F
import torch.distributed as dist
import math
from .KCL import KCL

__all__ =  ['Moco', 'KCL']

class Moco:
    def __init__(self, args, key_encoder, num_classes, dim, device='cuda', init_queue=None):
        """
        DDP-aware Per-class MoCo implementation
        
        Args:
            args: must have args.moco_k (queue capacity per class)
            key_encoder: momentum encoder (NOT wrapped with DDP)
            num_classes: number of classes
            dim: feature dimension
            device: device ('cuda' or 'cpu')
            init_queue: optional initial queue [num_classes, K, dim]
        """
        self.K = args.moco_k
        self.num_classes = num_classes
        self.dim = dim
        self.device = torch.device(device)
        
        # Key encoder 설정 (DDP로 감싸지 않음 - gradient 불필요)
        self.key_encoder = key_encoder.to(self.device)
        
        # Per-class queue 초기화: [num_classes, K, dim]
        if init_queue is None:
            self.queue = torch.randn(self.num_classes, self.K, self.dim, device=self.device)
            self.queue = F.normalize(self.queue, dim=2)
        else:
            self.queue = init_queue.to(self.device)
            print('init_queue', self.queue.shape)
            
        # Per-class 포인터: [num_classes]
        self.queue_ptr = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)
        
        # 기본 momentum
        self.m = 0.999
        
        # DDP 정보 저장 (동기화 판단용)
        self.is_distributed = dist.is_available() and dist.is_initialized()
        if self.is_distributed:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        self.args = args 
    # ... (다른 메서드들은 동일) ...

    @torch.no_grad()
    def _concat_all_gather(self, tensor):
        """
        DDP 환경에서 모든 프로세스의 텐서를 수집하여 concatenate
        
        Args:
            tensor: [local_batch_size, ...] 로컬 텐서
            
        Returns:
            gathered_tensor: [world_size * local_batch_size, ...] 전체 텐서
            
        동작원리:
        1. 모든 프로세스에서 동일한 크기의 텐서 리스트 생성
        2. all_gather로 각 프로세스의 텐서를 모든 프로세스에 복사
        3. 수집된 텐서들을 batch 차원으로 concatenate
        """
        if not self.is_distributed:
            return tensor
            
        # 모든 프로세스에서 텐서를 수집할 리스트 준비
        # world_size만큼의 동일한 shape 텐서들을 담을 리스트
        tensors_gather = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        
        # all_gather: 현재 프로세스의 tensor를 모든 프로세스의 tensors_gather에 복사
        # 완료 후: tensors_gather[i] = process_i의 tensor
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        
        # 수집된 텐서들을 batch 차원(dim=0)으로 연결
        # [local_bs, ...] * world_size -> [world_size*local_bs, ...]
        return torch.cat(tensors_gather, dim=0)

    @torch.no_grad()
    def encode_key(self, imgs, ldmks=None):
        """
        Encode key features from images
        """
        return F.normalize(self.key_encoder(imgs, keypoint=ldmks, features=True)[1], dim=1)

    @torch.no_grad()
    def enqueue_distributed(self, imgs, labels, ldmks=None):
        """
        DDP-aware enqueue: 모든 프로세스의 키/라벨을 수집 후 동기화된 큐 업데이트
        
        Args:
            imgs: [local_batch_size, ...] 로컬 배치 이미지
            labels: [local_batch_size] 로컬 배치 라벨
            ldmks: optional landmarks
            
        동작 과정:
        1. 로컬 키 계산 (각 GPU에서 서로 다른 배치)
        2. all_gather로 전체 키/라벨 수집
        3. 수집된 전체 데이터로 큐 업데이트 (모든 프로세스에서 동일)
        4. 결과: 모든 GPU의 큐가 동일한 상태 유지
        """
        # Step 1: 로컬 배치에서 키 계산
        # 각 프로세스(GPU)가 자신의 배치만 처리
        local_keys = self.encode_key(imgs,  ldmks)  # [local_batch_size, dim]
        local_labels = labels.to(self.device)              # [local_batch_size]
        
        # Step 2: DDP 환경에서 모든 프로세스의 키/라벨 수집
        if self.is_distributed:
            # 모든 GPU의 키를 하나로 모음
            # local_keys: [local_bs, dim] -> all_keys: [world_size*local_bs, dim]
            all_keys = self._concat_all_gather(local_keys)
            
            # 모든 GPU의 라벨을 하나로 모음  
            # local_labels: [local_bs] -> all_labels: [world_size*local_bs]
            all_labels = self._concat_all_gather(local_labels)
        else:
            # 단일 GPU 환경에서는 그대로 사용
            all_keys = local_keys
            all_labels = local_labels
            
        # Step 3: 수집된 전체 데이터로 큐 업데이트
        # 모든 프로세스가 동일한 all_keys, all_labels로 업데이트
        # -> 결과적으로 모든 프로세스의 큐가 동일해짐
        self._enqueue_keys_labels(all_keys, all_labels)
        
        # 참고: 모든 프로세스가 동일한 연산을 하므로 별도 동기화 불필요

    @torch.no_grad() 
    def _enqueue_keys_labels(self, keys, labels):
        """
        실제 큐 업데이트 로직 (분리된 내부 메서드)
        
        Args:
            keys: [total_batch_size, dim] 정규화된 키 특징
            labels: [total_batch_size] 클래스 라벨
            
        동작:
        - 클래스별로 키들을 그룹화
        - 각 클래스 큐에 FIFO 방식으로 추가 (원형 버퍼)
        - 포인터 업데이트로 다음 쓰기 위치 관리
        """
        keys = F.normalize(keys, dim=1)  # 안전용 재정규화
        
        # 클래스별 그룹화 후 큐 업데이트
        for c in labels.unique():
            c = int(c.item())                      # 정수 변환
            mask = (labels == c)                   # 클래스 c 마스크
            kc = keys[mask]                        # [n_c, dim] 클래스 c 키들
            n = kc.shape[0]                        # 클래스 c 샘플 수
            
            if n == 0:
                continue
                
            # 현재 포인터 위치에서 FIFO 삽입
            p = int(self.queue_ptr[c].item())      # 현재 쓰기 위치
            end = p + n                            # 쓰기 종료 위치
            
            if end <= self.K:
                # 큐 끝을 넘지 않는 경우: 연속 삽입
                self.queue[c, p:end, :] = kc
            else:
                # 큐 끝을 넘는 경우: wrap-around (원형 버퍼)
                first = self.K - p                 # 뒷부분에 들어갈 수
                self.queue[c, p:self.K, :] = kc[:first]        # 뒷부분 채우기
                self.queue[c, 0:(end % self.K), :] = kc[first:] # 앞부분 채우기
                
            # 포인터 업데이트 (원형)
            self.queue_ptr[c] = (p + n) % self.K

    @torch.no_grad()
    def enqueue(self, imgs, labels, ldmks=None):
        """
        Backward compatible enqueue method
        자동으로 DDP 환경 감지하여 적절한 동기화 수행
        """
        if self.is_distributed:
            self.enqueue_distributed(imgs, labels, ldmks)
        else:
            # 단일 GPU: 기존 방식
            keys = self.encode_key(imgs, ldmks)
            self._enqueue_keys_labels(keys, labels.to(self.device))

    @torch.no_grad()
    def get_k(self, labels, k):
        """
        각 샘플에 대해 해당 클래스 큐에서 정확히 k개 양성을 반환
        
        주의: DDP 환경에서는 모든 프로세스의 큐가 동일하므로
              각 프로세스에서 동일한 결과 반환 (동기화됨)
        """
        N = labels.numel()
        labels = labels.to(self.device)
        
        pos_feats = []
        pos_labels = []
        
        for i in range(N):
            c = int(labels[i].item())
            
            # With replacement로 정확히 k개 샘플링
            # 모든 프로세스에서 동일한 시드 사용하면 동일한 결과 (선택사항)
            indices = torch.randint(0, self.K, (k,), device=self.device)
            sampled_feats = self.queue[c, indices, :]                      # [k, dim]
            sampled_labels = torch.full((k,), c, dtype=labels.dtype, device=self.device)
            
            pos_feats.append(sampled_feats)
            pos_labels.append(sampled_labels)
        
        pos_feats = torch.stack(pos_feats, dim=0)   # [N, k, dim]
        pos_labels = torch.stack(pos_labels, dim=0) # [N, k]
        
        return pos_feats, pos_labels

    @torch.no_grad()
    def momentum_update(self, q_encoder):
        """
        Momentum 업데이트 (DDP 고려사항)
        
        주의사항:
        - q_encoder는 DDP로 감싸져 있음 (gradient 동기화됨)
        - key_encoder는 DDP 없이 직접 업데이트
        - 모든 프로세스에서 동일한 momentum 업데이트 수행
        """
        # DDP 모델의 경우 .module로 접근하여 실제 파라미터 얻기


        q_params = q_encoder.parameters()
        
        # 표준 momentum 업데이트: θ_k ← m*θ_k + (1-m)*θ_q
        for p_q, p_k in zip(q_params, self.key_encoder.parameters()):
            p_k.data.mul_(self.m).add_(p_q.data, alpha=1.0-self.m)

    # ... (나머지 메서드들은 이전과 동일) ...
