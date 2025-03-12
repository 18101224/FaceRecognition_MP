import cv2 as cv
import os

def init_quality_model(ckpt_path):
    model_path = os.path.join(ckpt_path,'model_live.yml')
    range_path = os.path.join(ckpt_path,'range_live.yml')
    return cv.quality.QualityBRISQUE_create(model_path,range_path)

def by_ml(model,img_path):
    img = cv.imread(img_path)
    return model.compute(img)[0]

