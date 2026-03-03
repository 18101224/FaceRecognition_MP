import torch 
from .Moco import Moco
import torch.nn.functional as F


class NCL:
    def __init__(self, args, key_encoder, num_classes, class_counts, dim, temperature=0.07):

        self.args = args 
        self.moco = Moco(
            args=args,
            key_encoder=key_encoder,
            num_classes=1,
            dim=dim,
            device=torch.device('cuda'),
            init_queue=None,
        )
        self.temperature = temperature
        # class_counts: [C] training set counts (for Balanced Softmax / logit adjustment)
        if not isinstance(class_counts, torch.Tensor) and class_counts is not None:
            class_counts = torch.tensor(class_counts)
        self.class_counts = class_counts.cuda() if class_counts is not None else None

        
    @torch.no_grad()
    def MA_inference(self,img_k, aligner=None):
        if aligner is not None:
            _,_,ldmk,_,_,_ = aligner(img_k) 
        else: 
            ldmk = None
        feature_k = self.moco.encode_key(img_k, ldmks=ldmk)
        return feature_k
        
    def compute_nil(self, logits, y):
        C = logits.size(-1)
        h = max(1, min(int(C * 0.3), C-1))   # hard negatives 개수

        logits_raw = logits

        # full loss: (원하면 adjustment 적용)
        logits_full = logits_raw
        if self.args.adjustment:
            logits_full = logits_full + torch.log(self.class_counts.view(1,-1).to(logits.device))
        ce_all = F.cross_entropy(logits_full, y)

        # hard mining: raw에서 GT 제외하고 neg h개 뽑기
        scores = logits_raw.detach()
        scores.scatter_(1, y.view(-1,1), float("-inf"))
        topk_neg = torch.topk(scores, k=h, dim=1, largest=True, sorted=False).indices

        mask = torch.zeros_like(logits_raw, dtype=torch.bool)
        mask.scatter_(1, topk_neg, True)
        mask.scatter_(1, y.view(-1,1), True)   # GT 포함 => 총 h+1개

        logits_hard = logits_raw.masked_fill(~mask, float("-inf"))
        if self.args.adjustment:
            logits_hard = logits_hard + torch.log(self.class_counts.view(1,-1).to(logits.device))
        ce_hard = F.cross_entropy(logits_hard, y)

        return ce_all + ce_hard

    def __call__(self, logits, features, y, aligner, positive_pair, **kwargs):
        '''
        logits : bs, C
        features : bs, D
        y : bs
        '''
        # NIL part (classification): all + hard-mined
        ce_loss = self.compute_nil(logits,y)

        k = self.MA_inference(positive_pair, aligner)

        positive_logit = torch.einsum('nd,nd->n', [features, k]).unsqueeze(-1) # bs, 1
        negative_logit = torch.einsum('nd,kd->nk', [features, self.moco.get_queue(class_idx=0)]) # bs, K
        logits = torch.cat([positive_logit, negative_logit], dim=1) # bs, K+1
        logits /= self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        cl_loss = F.cross_entropy(logits, labels)
        return ce_loss, cl_loss, k

    @torch.no_grad()
    def momentum_update(self, model):
        self.moco.momentum_update(model)
    

    def enqueue(self,k):
        # k: [bs, dim] key embeddings
        self.moco.enqueue_embeddings(k, class_idx=0)