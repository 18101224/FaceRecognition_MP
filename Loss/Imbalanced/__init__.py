import torch
from .BCL import BCLLoss
from .ECE import get_angle_loss, weight_scheduling, ECELoss
from .SLERP import spherical_frechet_mean


__all__ = ['BCLLoss', 'get_angle_loss', 'weight_scheduling', 'ECELoss', 'calculate_class_mean']




def calculate_class_mean(dataloader, model, device, use_gradient=False):
    """
    Calculate class means from dataloader.

    Args:
        dataloader: DataLoader containing the data
        model: Model to extract features
        device: Device to run computations on
        use_gradient: If True, gradients will be computed; if False, no_grad context is used
    """
    if use_gradient:
        # Gradients will be computed
        context = torch.enable_grad()
        
    else:
        # No gradients computed (default behavior)
        context = torch.no_grad()
    features = []
    labels = []
    with context:
        for image, label in dataloader : 
            image = image.to(device)
            label = label.to(device)
            feature, _, _ = model(image, features=True)
            features.append(feature)
            labels.append(label)
    
    num_classes=len(dataloader.dataset.img_num_list)
    class_centers = []
    for i in range(num_classes):
        class_features = features[labels==i]
        class_mean = spherical_frechet_mean(class_features)
        class_centers.append(class_mean)
    class_centers = torch.stack(class_centers,dim=0)
    return class_centers


