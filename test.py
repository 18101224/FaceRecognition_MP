import torch 
from tqdm import tqdm 
from functools import partial
import numpy as np 
import matplotlib.pyplot as plt 

dim = 9
n_c = 10 

weight = torch.randn((n_c,dim),requires_grad=True,device=torch.device('mps'))
norm = partial(torch.nn.functional.normalize, dim=1, p=2)
opt = torch.optim.SGD([weight], lr=100)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10000, eta_min=0.000001)
means = []
stds = []

for i in tqdm(range(10000)):
    opt.zero_grad()
    w = norm(weight)
    sim = w@w.T
    indices = torch.triu_indices(n_c, n_c, offset=1)
    sims = sim[indices[0],indices[1]]
    s = ((-1-sim.reshape(-1))**2).mean()
    std = torch.abs(sim.reshape(-1)-s).mean()
    means.append(s.detach().cpu().item())
    stds.append(std.detach().cpu().item())
    loss = s+std*2
    loss.backward()
    opt.step()
    scheduler.step()
    if i%100==0:
        with torch.no_grad():
            sim = norm(weight) @ norm(weight).T
            # Clamp values to valid range for arccos to avoid numerical errors
            sim_clamped = sim.clamp(-1.0, 1.0)
            angles = torch.acos(sim_clamped) * 180.0 / torch.pi  # Convert radians to degrees
            indices = torch.triu_indices(n_c, n_c, offset=1)
            upper_tri = angles[indices[0],indices[1]]
            upper_tri_mean = upper_tri.mean().item()
            upper_tri_std = upper_tri.std().item()
            print(f'upper triangle mean (no diag): {upper_tri_mean:.4f}, std: {upper_tri_std:.4f}')

means = np.array(means)
stds = np.array(stds)

plt.figure(figsize=(8, 6))
plt.plot(means, label='mean')
plt.plot(stds, label='std')
plt.legend()
plt.title('mean and std')
plt.show()

# INSERT_YOUR_CODE
with torch.no_grad():
    sim = norm(weight) @ norm(weight).T
    sim_clamped = sim.clamp(-1.0, 1.0)
    angles = torch.acos(sim_clamped) * 180.0 / torch.pi  # Convert radians to degrees

plt.figure(figsize=(8, 6))
im = plt.imshow(angles.cpu().numpy(), cmap='coolwarm', vmin=80, vmax=100)
plt.colorbar(im, label='Angle (degrees)')
plt.title('Pairwise Angle Matrix (Centered at 90°, Range 80-100°)')
plt.xlabel('Class Index')
plt.ylabel('Class Index')
plt.tight_layout()
plt.savefig(f'{dim}.png')
plt.show()



# import torch 
# from torch import nn 
# from torch.nn import functional 

# w = torch.randn((100,64))
# w_normed = functional.normalize(w,dim=1,p=2)
# print(w_normed[0].norm(p=2))
# w_T = w.T 
# w_T_normed = functional.normalize(w_T,dim=0,p=2)
# print(w_T_normed[:,0].norm(p=2))