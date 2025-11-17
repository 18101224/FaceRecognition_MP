import torch 

class SCN:

    def get_indices(self, alpha):
        sorted_indices = torch.argsort(alpha, descending=True)
        return sorted_indices

    def __call__(self, logits, features, y, **kwargs):
        '''
        logits : original logits 
        features : alpha  ( bs,)
        
        '''
        bs = y.shape[0]
        M = int(bs*0.7)
        ce_loss = torch.nn.functional.cross_entropy(logits*features.reshape(bs,1), y)
        indices = self.get_indices(features)
        highs = indices[:M]
        lows = indices[M:]
        alpha_highs = (1/M)*(features[highs].sum())
        alpha_lows = 1/(bs-M)*(features[lows].sum())
        rr = max(torch.tensor(0, device=features.device), 0.15-(alpha_highs - alpha_lows))
        return ce_loss, rr, None 
