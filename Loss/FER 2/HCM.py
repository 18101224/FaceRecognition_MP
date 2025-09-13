import torch 
import numpy as np 

class HCM:
    def __init__(self, args, train_set, weight=None, ):
        self.bank = self.calc_sims(weight)
        self.classification = args.classification 
        self.label = torch.tensor([1,0]+[3,2]+[5,4]).reshape(-1).cuda()
        self.cl_weight = args.cl_weight
        self.args = args
        self.cls_counts = train_set.get_img_num_per_cls()

    @torch.no_grad()
    def calc_sims(self,weight):
        if weight is None:
            return 
        sims = (weight.T @ weight ).clone()
        sims.fill_diagonal_(-np.inf)
        sorted_classes = sims.sort(dim=-1,descending=True)[1]
        hard_indices = sorted_classes[:,0].reshape(-1)
        cls_counts = torch.tensor(self.cls_counts).sort(dim=-1,descending=True)[-1].reshape(-1)
        bank = cls_counts.unsqueeze(0).repeat(7,1)
        bank[:,0] = hard_indices
        for i in range(7):
            if bank[i,0] == bank[i,1]:
                bank[i,1] = cls_counts[2] if not i==2 else cls_counts[3]
        return bank.cuda()
    
    def calc_mining(self, to_cl, bs):
        to_cl = to_cl.unsqueeze(1) # (bs+bs+k*bs+k*bs, 1, dim)
        anchor, positive, hard_neg, head_neg = torch.split(to_cl, [bs,bs,self.args.k*bs,self.args.k*bs], dim=0) 
        head_neg = head_neg.reshape(bs, self.args.k, -1) # (bs, k, dim)
        hard_neg = hard_neg.reshape(bs, self.args.k, -1) # (bs, k, dim)
        pairs = torch.cat([anchor, positive, hard_neg, head_neg], dim=1) # (bs, 1+1+k+k, dim)
        sims = pairs @ pairs.transpose(-1,-2) # (bs, 1+1+k+k, 1+1+k+k) # label = [true, true, false*k, false*k] 

        loss = torch.nn.functional.cross_entropy(sims.reshape(-1,sims.shape[-1]), self.label.repeat(bs))
        return loss 
        
    def __call__(self, images, labels, sampler, model, aligner=None,):
        bs, c, h, w = images.shape
        # Ensure bs is a Python int for indexing/slicing
        positive, y_p = sampler.sample_pairs(labels, k=1, num_workers=self.args.num_workers)
        bank = self.calc_sims(model.get_kernel() if self.args.world_size==1 else model.module.get_kernel()) if getattr(self,'bank',None) is self.bank else self.bank
        hard_neg, y_n1 = sampler.sample_pairs(bank[labels,0], k=self.args.k, num_workers=self.args.num_workers)
        head_neg, y_n2 = sampler.sample_pairs(bank[labels,1], k=self.args.k, num_workers=self.args.num_workers)
        images = torch.cat([images, positive, hard_neg.reshape(-1,c,h,w), head_neg.reshape(-1,c,h,w)], dim=0)
        if aligner is not None:
            with torch.no_grad():
                _,_,keypoint,_,_,_ = aligner(images)
        feature, logit, centers = model(images, keypoint=keypoint if aligner is not None else None, features=True)
        to_cl = logit if self.classification else feature
        hcm_loss = self.calc_mining(to_cl, bs)
        ce_loss = torch.nn.functional.cross_entropy(logit[:bs], labels)

        return hcm_loss*self.cl_weight + ce_loss, logit[:bs]