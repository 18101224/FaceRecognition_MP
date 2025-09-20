#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Requirements: pip install -U insightface opencv-python onnxruntime

import os
import glob
from typing import List, Tuple
from dataclasses import dataclass

import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ---------------------------
# Config: ArcFace 5-point template (112x112)
# ---------------------------
ARCFACE_5PTS_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)  # widely used 5pt template for alignment to 112x112[web:24][web:43]

# ---------------------------
# Detection structures/utilities
# ---------------------------
@dataclass
class DetectedFace:
    bbox: np.ndarray        # (4,) x1,y1,x2,y2
    kps: np.ndarray | None  # (5,2) or None
    score: float

class FaceDetector:
    def __init__(self, det_size: Tuple[int,int]=(640,640),
                 provider: str='CPUExecutionProvider',
                 name: str='antelopev2'):
        # InsightFace FaceAnalysis wraps SCRFD/RetinaFace etc.
        self.app = FaceAnalysis(name=name, providers=[provider])
        self.app.prepare(ctx_id=0, det_size=det_size)  # auto-downloads models[web:15][web:13]

    def detect(self, img: np.ndarray) -> List[DetectedFace]:
        faces = self.app.get(img)  # returns objects with .bbox, .kps, .det_score[web:15]
        out: List[DetectedFace] = []
        for f in faces:
            kps = getattr(f, "kps", None)
            out.append(DetectedFace(
                bbox=f.bbox.astype(np.float32),
                kps=kps.astype(np.float32) if kps is not None else None,
                score=float(getattr(f, "det_score", 1.0))
            ))
        return out

    @staticmethod
    def pick_largest(faces: List[DetectedFace]) -> int:
        if not faces:
            return -1
        areas = [(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]) for f in faces]
        return int(np.argmax(areas))

# ---------------------------
# Alignment / cropping
# ---------------------------
def align_by_5pts(img: np.ndarray, kps: np.ndarray, out_size=(112,112)) -> np.ndarray:
    """
    Align with 5-point landmarks using similarity transform to ArcFace template.
    InsightFace/RetinaFace 5pts order is typically [left_eye, right_eye, nose, left_mouth, right_mouth] or similar; 
    using estimateAffinePartial2D is robust as long as point correspondence is consistent.[web:42][web:30]
    """
    assert kps.shape == (5,2), "kps must be (5,2)"
    src = ARCFACE_5PTS_112.astype(np.float32)
    dst = kps.astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(dst, src, method=cv2.LMEDS)
    aligned = cv2.warpAffine(img, M, out_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return aligned

def crop_with_bbox(img: np.ndarray, bbox: np.ndarray, margin: float=0.2) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)
    bw, bh = x2-x1, y2-y1
    mx, my = int(bw*margin), int(bh*margin)
    x1, y1 = max(0, x1-mx), max(0, y1-my)
    x2, y2 = min(w, x2+mx), min(h, y2+my)
    return img[y1:y2, x1:x2]

# ---------------------------
# IO helpers
# ---------------------------
def list_images(path: str):
    if os.path.isdir(path):
        files = []
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
            files.extend(glob.glob(os.path.join(path, ext)))
        return sorted(files)
    return [path]

def make_output_path(input_path: str, output_dir: str) -> str:
    base = os.path.basename(input_path)
    stem, ext = os.path.splitext(base)
    if ext.lower() not in [".jpg",".jpeg",".png",".bmp",".webp"]:
        ext = ".png"
    return os.path.join(output_dir, f"{stem}_aligned{ext}")

def imread(path: str):
    return cv2.imread(path)

def imsave(path: str, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)

# ---------------------------
# Public API
# ---------------------------
def process(input_path: str, output_dir: str,
            align: bool=True) -> None:
    """
    Detects the largest face in each image, aligns with 5 landmarks if available,
    and saves as <orig>_aligned.<ext> to output_dir.[web:15][web:13][web:24]
    """
    os.makedirs(output_dir, exist_ok=True)
    global detector


    for ip in list_images(input_path):
        img = imread(ip)
        if img is None:
            print(f"Skip (cannot read): {ip}")
            continue
        faces = detector.detect(img)
        if not faces:
            print(f"No face: {ip}")
            continue

        idx = detector.pick_largest(faces)
        face = faces[idx]

        if align and face.kps is not None and face.kps.shape == (5,2):
            out_img = align_by_5pts(img, face.kps, out_size=(112,112))
        else:
            out_img = crop_with_bbox(img, face.bbox, margin=0.2)

        out_path = make_output_path(ip, output_dir)
        imsave(out_path, out_img)
        print(f"Saved: {out_path}")

# ---------------------------
# CLI usage example
# ---------------------------
if __name__ == "__main__":
    # Example:
    # process("path/to/image_or_dir", "path/to/save_dir")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-align", action="store_true", help="disable 5-point alignment (bbox crop only)")
    parser.add_argument("--detector", type=str, default="scrfd", help="scrfd or retinaface")
    parser.add_argument("--det-size", type=int, nargs=2, default=(640,640))
    parser.add_argument("--provider", type=str, default="CPUExecutionProvider")
    
    from glob import glob 
    from tqdm import tqdm 

    detector = FaceDetector(det_size=(640,640), provider="CUDAExecutionProvider", name="antelopev2")
    s_root = '../data/RAF-DB_original'
    output_dir = './RAF-DB_oa'
    os.makedirs(output_dir, exist_ok=True)
    for post in ['train', 'valid']:
        for i in range(1,8):

            temp_root = f'{s_root}/{post}/{i}'
            temp_output_dir = f'{output_dir}/{post}/{i}'
            paths = glob(f'{temp_root}/*')
            print(len(paths))
            for path in tqdm(paths, desc=f"Processing {post} {i}") : 
                file_name = path.split('/')[-1].split('.')[0]
                process(path, output_dir=temp_output_dir,align=True)
    
