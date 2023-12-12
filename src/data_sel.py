import torch.utils.data as td
from src.dataloader import DatasetAnimals, pad, TIL, DatasetBirds, Nec, ColoredMNIST, OODBird, CelebA, DDI
import torchvision as tv
import numpy as np
import sklearn.model_selection as skms
import pdb
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import random

def calc_samples(X, y):
    maj_zero = 0
    maj_one = 0
    min_zero = 0 
    min_one = 0
    
    if X.size(0)!= 0:
        for i in range(X.size(0)):
            if y[i] == 0 and X[i, 0, :, : ].mean() == 0:
                min_zero+=1
            elif y[i] == 0 and X[i, 1, :, : ].mean() == 0:
                maj_zero+=1
            elif y[i] == 1 and X[i, 0, :, : ].mean() == 0:
                maj_one+=1
            elif y[i] == 1 and X[i, 1, :, : ].mean() == 0:
                min_one+=1
        
        print(f'Majority samples - 0: {100*(maj_zero)/(maj_zero+min_zero):2f} Majority samples - 1: {100*(maj_one)/(maj_one+min_one):2f} Majority samples total: {100*(maj_one+maj_zero)/(maj_one+min_one+maj_zero+min_zero):2f}\n')
 
def selector(args):
    # pad images to 500 pixels
    max_padding = tv.transforms.Lambda(lambda x: pad(x, fill=fill))
    random.seed(0)

    fill = tuple(map(lambda x: int(round(x * 256)), (0.485, 0.456, 0.406)))
    transforms_train = tv.transforms.Compose([
        max_padding,
        tv.transforms.RandomOrder([
        tv.transforms.RandomCrop((375, 375)),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.RandomVerticalFlip()
        ]),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transforms_eval = tv.transforms.Compose([
        max_padding,
        tv.transforms.CenterCrop((375, 375)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if args.dset == 'animals':
        ds_train = DatasetAnimals(args.data_dir, transform=transforms_train, train=True)
        ds_val = DatasetAnimals(args.data_dir, transform=transforms_eval, train=True)
        ds_test = DatasetAnimals(args.data_dir, transform=transforms_eval, train=False)

    if args.dset == 'til':
        ds_train = TIL(args.data_dir,args, transform=transforms_train, train=True)
        ds_val = TIL(args.data_dir, args,transform=transforms_eval, train=True)
        ds_test = TIL(args.data_dir, args,transform=transforms_eval, train=False)

    if args.dset == 'til_necrosis':
        ds_train = Nec(args.data_dir,args, transform=transforms_train, train=True)
        ds_val = Nec(args.data_dir, args,transform=transforms_eval, train=True)
        ds_test = Nec(args.data_dir, args, transform=transforms_eval, train=False)

    if args.dset == 'birds':
        ds_train = DatasetBirds(args.data_dir,args, transform=transforms_train, train=True)
        ds_val = DatasetBirds(args.data_dir, args, transform=transforms_eval, train=True)
        ds_test = DatasetBirds(args.data_dir,args, transform=transforms_eval, train=False)
    
    if args.dset == 'birds-ood':
        # pdb.set_trace()
        ds_train = OODBird(args.data_dir, transform=transforms_train)
        ds_val = OODBird(args.data_dir, transform=transforms_eval, env = 'in_test')
        ds_test = OODBird(args.data_dir, transform=transforms_eval, env = 'in_test')
    
    if args.dset == 'celeba':
        ds_train = CelebA(args.data_dir, env='train', transform = transforms_train)
        ds_test_in = CelebA(args.data_dir, env='in_test', transform = transforms_train)
        ds_test_out = CelebA(args.data_dir, env='out_test', transform = transforms_train)

    train_params = {'batch_size': args.batch_size}
    test_params =  {'batch_size': args.batch_size}
    splits = skms.StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
    idx_train, idx_val = next(splits.split(np.zeros(len(ds_train)), ds_train.targets))
    # instantiate data loaders
    trainset = td.DataLoader(
    dataset=ds_train,
    sampler=td.SubsetRandomSampler(idx_train),
    **train_params
    )
    validset = td.DataLoader(
    dataset=ds_val,
    sampler=td.SubsetRandomSampler(idx_val),
    **train_params
    )

    test_loader = td.DataLoader(dataset=ds_test, **test_params)
    print(len(ds_train), len(ds_val), len(ds_test))

    return trainset, validset, test_loader