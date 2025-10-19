# -*- coding: utf-8 -*-
import torch
from .Moco import Moco
from torch import nn 


class KCL(nn.Module):
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
                 args, 
                 key_encoder, 
                 dim, 
                 temperature: float = 0.07,
                 init_queue=None
                 ):
        super().__init__()
        self.tau = float(temperature)                              # InfoNCE 온도 파라미터                             # 손실 축약 방식
        self.moco = Moco(args, key_encoder=key_encoder, num_classes=args.num_classes,
         dim=dim, device=torch.device('cuda'), init_queue=init_queue) 
        self.args = args 
        self.ce_loss = torch.nn.CrossEntropyLoss().to(torch.device('cuda'))

    def forward(self,
                logits  , # [N, C]
                features ,      # [N, D] 앵커 임베딩
                y , 
                weight ,
                centers  ,
                positive_pair  = None ,
                **kwargs        
                ):           # scalar 손실값
        
        N, D = features.shape                                      # N: 배치 크기, D: 임베딩 차원
  # [N, D] 정규화된 앵커

        ce_loss = self.ce_loss(logits, y)

        if weight is not None :
            features = torch.cat([features, weight], dim=0)
        if centers is not None :
            features = torch.cat([features, centers], dim=0)

        if weight is not None or centers is not None :
            y = torch.cat([y, torch.arange(self.args.num_classes, device=torch.device('cuda')).repeat(2 if self.args.utilize_class_centers else 1)], dim=0)

        if positive_pair is None :
            positive_pair = self.moco.get_k(y,k=self.args.kcl_k)
        # 입력 형태 판별: 고정 길이 vs 가변 길이

        cl_loss = self.compute_fixed(q=features, k=positive_pair, y=y)

        return ce_loss, cl_loss, positive_pair

    @torch.no_grad()
    def momentum_update(self, model):
        self.moco.momentum_update(model)
    
    @torch.no_grad()
    def enqueue(self, img, label, ldmk):
        self.moco.enqueue(img, label, ldmk)


    def compute_fixed(self,q,k,y):
        '''
        query, key , label 
        '''
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

        pos_sims = torch.cat([k_sims, batch_pos_sims],dim=1 ) # N,k+N 
        y_counts = y.bincount().to(q.device) # C 
        neg_sims = batch_sims*neg_batches # N,N * N,N -> N,N 
        neg_sims = neg_sims.masked_fill(neg_sims == 0, float('-inf'))  # N, N 

        num = torch.logsumexp(pos_sims, dim=1) # N 
        den = torch.log(torch.sum(torch.exp(neg_sims) / y_counts[y].reshape(1,-1),dim=1).reshape(N,))

        loss = -(num - den) / same_class_counts

        return loss.mean()

