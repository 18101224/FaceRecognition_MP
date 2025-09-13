from torch import nn 
import torch 


def Attn_QCS_SD(anchor, positive_squences, negative_squences, k):
    B,N,C = anchor.shape
    k = k + 1 

    similarity_maps = []

    for pos in positive_squences : 
        qk = pos[0]
        D_2x1 = torch.cdist(qk, anchor, p=2)
        D_2x1 = D_2x1 - torch.min(D_2x1)
        S_2x1 = torch.max(D_2x1) - D_2x1
        similarity_maps.append(S_2x1)

    distance_maps = []
    for neg in negative_squences : 
        qk = neg[0]
        D_2x1 = torch.cdist(qk, anchor, p=2)
        D_2x1 = D_2x1 - torch.min(D_2x1)
        S_2x1 = torch.max(D_2x1) - D_2x1
        similarity_maps.append(S_2x1)

    return similarity_maps

class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim * 2)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=int(embed_dim * 4), act_layer=nn.GELU, drop=0)
        self.theta = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, anchor, positives, negatives):
        B, N, C = anchor.shape

        anchor_qk, anchor_v = self.proj(anchor).reshape(B,N,2,C).permute(2,0,1,3)

        positive_sequences = []
        for pos  in positives : 
            qk,v = self.proj(pos).reshape(B,N,2,C).permute(2,0,1,3)
            positive_sequences.append((qk,v))

        negative_sequences = []
        for neg in negatives : 
            qk,v = self.proj(neg).reshape(B,N,2,C).permute(2,0,1,3)
            negative_sequences.append((qk,v))
        
        k = torch.tanh(self.theta)
        