import torch
import torch.utils.data as td
import torchvision.transforms as transforms
import numpy as np
import sklearn.model_selection as skms
import random
from src.data.dataloader import DatasetAnimals, pad, TIL, DatasetBirds, Nec, OODBird, CelebA

def calc_samples(X, y):
    conditions = [(y == 0) & (X[:, 0, :, :].mean(dim=(1, 2)) == 0),
                  (y == 0) & (X[:, 1, :, :].mean(dim=(1, 2)) == 0),
                  (y == 1) & (X[:, 0, :, :].mean(dim=(1, 2)) == 0),
                  (y == 1) & (X[:, 1, :, :].mean(dim=(1, 2)) == 0)]
    counts = [torch.sum(cond).item() for cond in conditions]
    min_zero, maj_zero, maj_one, min_one = counts
    total = sum(counts)
    print(f'Majority samples - 0: {100 * maj_zero / total:.2f}% '
          f'Majority samples - 1: {100 * maj_one / total:.2f}% '
          f'Majority samples total: {100 * (maj_one + maj_zero) / total:.2f}%\n')

def get_transforms():
    fill = tuple(map(lambda x: int(round(x * 256)), (0.485, 0.456, 0.406)))
    max_padding = transforms.Lambda(lambda x: pad(x, fill=fill))
    transforms_train = transforms.Compose([
        max_padding,
        transforms.RandomOrder([
            transforms.RandomCrop((375, 375)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transforms_eval = transforms.Compose([
        max_padding,
        transforms.CenterCrop((375, 375)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transforms_train, transforms_eval

def load_dataset(dataset_class, args, transforms, train=True):
    if dataset_class in [CelebA]:
        return dataset_class(args.data_dir, transform=transforms, env='train' if train else 'in_test')
    return dataset_class(args.data_dir, args, transform=transforms, train=train)

def selector(args):
    transforms_train, transforms_eval = get_transforms()

    dataset_mapping = {
        'animals': DatasetAnimals,
        'til': TIL,
        'til_necrosis': Nec,
        'birds': DatasetBirds,
        'birds-ood': OODBird,
        'celeba': CelebA
    }

    dataset_class = dataset_mapping.get(args.dset)
    if not dataset_class:
        raise ValueError(f"Dataset {args.dset} not recognized.")

    ds_train = load_dataset(dataset_class, args, transforms_train, train=True)
    ds_val = load_dataset(dataset_class, args, transforms_eval, train=True)
    ds_test = load_dataset(dataset_class, args, transforms_eval, train=False)

    train_params = {'batch_size': args.batch_size}
    test_params = {'batch_size': args.batch_size}
    splits = skms.StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
    idx_train, idx_val = next(splits.split(np.zeros(len(ds_train)), ds_train.targets))

    trainset = td.DataLoader(dataset=ds_train, sampler=td.SubsetRandomSampler(idx_train), **train_params)
    validset = td.DataLoader(dataset=ds_val, sampler=td.SubsetRandomSampler(idx_val), **train_params)
    test_loader = td.DataLoader(dataset=ds_test, **test_params)

    print(len(ds_train), len(ds_val), len(ds_test))

    return trainset, validset, test_loader
