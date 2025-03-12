import torch

def compute_instance_adaloss(gen_logit, tr_logit, alpha):
    I = torch.argmax(gen_logit,dim=1) == torch.argmax(tr_logit,dim=1) # 1
    return (1+alpha*(I.float()))

def quality_ada(pred,label,img_quality):
    pred = torch.argmax(pred) == label
    return (1+pred*torch.nn.functional.sigmoid(img_quality))

def margine_loss(cos,m,label,pred):
    thetas = torch.arccos(cos)
