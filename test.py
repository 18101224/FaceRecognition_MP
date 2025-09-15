from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import MTCNN
import torch
import os

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

input_path = 'face.png'
img = Image.open(input_path).convert('RGB')
img = img.resize((160, 160), Image.Resampling.BILINEAR)

# Use detect to obtain landmarks (N, 5, 2) with MPS->CPU fallback for pooling bug
try:
    boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
except RuntimeError as e:
    if 'Adaptive pool MPS' in str(e):
        print('Hit MPS adaptive pooling bug; retrying detection on CPU...')
        mtcnn_cpu = MTCNN(keep_all=True, device=torch.device('cpu'))
        boxes, probs, landmarks = mtcnn_cpu.detect(img, landmarks=True)
    else:
        raise

print(type(landmarks))
if landmarks is None:
    print('No faces detected, nothing to draw.')


else:
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    radius = 3
    for face_idx, pts in enumerate(landmarks):
        # pts: (5, 2) in pixel coords (x, y)
        for li, (x, y) in enumerate(pts):
            x0, y0 = x - radius, y - radius
            x1, y1 = x + radius, y + radius
            draw.ellipse([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
            label = str(li)
            # Small offset for readability
            tx, ty = x + radius + 1, y - radius - 1
            if font is not None:
                draw.text((tx, ty), label, fill=(255, 255, 0), font=font)
            else:
                draw.text((tx, ty), label, fill=(255, 255, 0))

    base, ext = os.path.splitext(input_path)
    out_path = f"{base}_landmarks{ext or '.jpg'}"
    img.save(out_path)
    print(f'Saved annotated image to {out_path}')

# landmarks shape: (N, 5, 2)