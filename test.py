from aligners import get_aligner 
from PIL import Image 
from torchvision import transforms 
from dataset import get_transform 
from argparse import Namespace
import torch 
import cv2
import numpy as np
device = torch.device('cuda')
aligner = get_aligner('checkpoint/adaface_vit_base_kprpe_webface12m').to(device)

args = {
    'dataset_name': 'RAF-DB', 'model_type': 'kprpe'
}
args = Namespace(**args)
transform = get_transform(args, train=False)
img = Image.open('Profile.png')
if img.mode != 'RGB':
    img = img.convert('RGB')
img = transform(img)
img = img.unsqueeze(0).to(device)

_,_,ldmk,_,_,_ = aligner(img)
print(ldmk)

ldmk = ldmk.squeeze(0).cpu().numpy()

c, h, w = img[0].shape
img = ((img*0.5)+0.5)*255
img = img[0].detach().cpu().numpy().transpose(1,2,0).astype(np.uint8)
print(img.dtype)
print(img.shape)
print(type(img))
img = np.ascontiguousarray(img)
for y,x in ldmk : 
    cv2.circle(img, (int(x * w), int(y * h)), 1, (0, 0, 255), 4)
cv2.imwrite('test.jpg', img)
