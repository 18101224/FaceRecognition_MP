from .sampler import ImbalancedDatasetSampler
from .sampler_wrapper import DistributedSamplerWrapper
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .clothing1m import Clothing1m_KFOLD, Clothing1m
from .fer import FER_KFOLD, FER
from .transform import get_clothing1m_transforms, get_fer_transforms 


ds_kfolds = {
    'clothing1m': Clothing1m_KFOLD,
    'AffectNet': FER_KFOLD
}
ds = {
    'clothing1m': Clothing1m,
    'AffectNet': FER
}
tf = {
    'clothing1m': get_clothing1m_transforms,
    'AffectNet': get_fer_transforms
}

def get_noise_dataset(args, train=True):
    dataset = ds[args.dataset_name]
    transform = tf[args.dataset_name]
    return dataset(args=args, transform=transform(train=train), train=train)


def get_kfolds(n_folds, args, random_seed):
    train_sets = []
    valid_sets = []
    dataset = ds_kfolds[args.dataset_name]
    transform = tf[args.dataset_name]
    print(dataset)
    for fold_idx in range(n_folds):
        train_sets.append(dataset(args=args, n_folds=n_folds, fold_idx=fold_idx, transform=transform(train=True), train=True, random_seed=random_seed))
        valid_sets.append(dataset(args=args, n_folds=n_folds, fold_idx=fold_idx, transform=transform(train=False), train=False, random_seed=random_seed))
    return train_sets, valid_sets


def get_loaders(args, train_sets, valid_sets, use_ddp=False):
    train_samplers = []
    valid_samplers = []
    for train_set, valid_set in zip(train_sets, valid_sets):
        train_sampler = ImbalancedDatasetSampler(train_set,labels=train_set.labels)
        if use_ddp:
            train_sampler = DistributedSamplerWrapper(train_sampler,shuffle=True)
            valid_sampler = DistributedSampler(valid_set,shuffle=False)
        else:
            valid_sampler = None
        train_samplers.append(train_sampler)
        valid_samplers.append(valid_sampler)
    train_loaders = [DataLoader(train_set,batch_size=args.batch_size,shuffle=False,sampler=train_sampler) for train_set, train_sampler in zip(train_sets, train_samplers)]
    valid_loaders = [DataLoader(valid_set,batch_size=args.batch_size,shuffle=False,sampler=valid_sampler) for valid_set, valid_sampler in zip(valid_sets, valid_samplers)]
    return train_loaders, valid_loaders, train_samplers, valid_samplers
