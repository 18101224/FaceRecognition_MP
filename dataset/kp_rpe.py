from torchvision import transforms

def get_kprpe_transform_valid():
    return transforms.Compose([
        transforms.Resize((112,112)),transforms.ToTensor(),transforms.Normalize([0.5]*3,[0.5]*3)
    ])

def get_kprpe_transform_train():
    return transforms.Compose([
        transforms.Resize((112,112)),
        transforms.RandomResizedCrop(112,scale=(0.5,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
