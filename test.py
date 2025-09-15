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
img = Image.open('face.png')
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

for idx, (x,y) in enumerate(ldmk) : 
    px = int(x * w)
    py = int(y * h)
    # Draw point
    cv2.circle(img, (px, py), 1, (0, 0, 255), 4)
    # Add index label with better visibility
    cv2.putText(img, str(idx), (px+5, py-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)  # black outline
    cv2.putText(img, str(idx), (px+5, py-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)  # white text
cv2.imwrite('test.jpg', img)