import torch 
import numpy as np 
from tqdm import tqdm 

device =torch.device('cpu')
weight = torch.randn((8,512),requires_grad=True, device=device)
opt = torch.optim.SGD([weight], lr=1e-6)

for i in tqdm(range(60000)):
    opt.zero_grad()
    w = torch.nn.functional.normalize(weight, dim=1, eps=1e-8)
    sims = w @ w.T 
    indices = torch.triu_indices(sims.shape[0], sims.shape[1], offset=1)
    elements = sims[indices[0], indices[1]].reshape(-1)
    loss = -(elements.mean() + elements.std() *0.1)*1000
    loss.backward()
    opt.step()

sims = w @ w.T 
print(torch.arccos(sims)*180/np.pi)


print(np.arccos(-1/7)*180/np.pi)