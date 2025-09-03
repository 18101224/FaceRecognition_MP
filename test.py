from models import ImbalancedModel
from dataset import FER, get_transform, ImbalancedDatasetSampler
import torch
from argparse import Namespace
from torch.utils.data import DataLoader
from opt import SAM


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps')
model = ImbalancedModel(num_classes=7, model_type='ir50', feature_branch=False)
model = model.to(device)
args = Namespace(**{'dataset_path':'/Users/seominjae/data/RAF-DB','dataset_name':'RAF-DB','loss':'CE','model_type':'ir50'})
train_transfrom = get_transform(args,train=True)
valid_transfrom = get_transform(args,train=False)
train_dataset = FER(args,train_transfrom,train=True, idx=False)
valid_dataset = FER(args,valid_transfrom,train=False, idx=False)
train_loader = DataLoader(train_dataset,batch_size=128,shuffle=False, sampler=ImbalancedDatasetSampler(train_dataset,labels=train_dataset.labels))
valid_loader = DataLoader(valid_dataset,batch_size=128,shuffle=False)

opt = SAM(model.parameters(),torch.optim.AdamW,lr=0.00001,weight_decay=0.001,adaptive=True)
for epoch in range(10):
    for img , label in train_loader : 
        img = img.to(device)
        label = label.to(device)
        pred = model(img, features=False)
        loss = torch.nn.functional.cross_entropy(pred, label)
        loss.backward()
        opt.first_step(zero_grad=True)
        pred = model(img, features=False)
        loss = torch.nn.functional.cross_entropy(pred, label)
        loss.backward()
        opt.second_step(zero_grad=True)
