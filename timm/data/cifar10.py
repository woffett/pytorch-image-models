'''
An addition to the timm library that includes compatibility with CIFAR10
and its standard augmentations

Adapted from:
https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
'''

import torch
import torchvision
import torchvision.transforms as transforms

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

TEST_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

def cifar10_loader(args, train=False):
    if train:
        transform = TRAIN_TRANSFORMS
        shuffle = True
    else:
        transform = TEST_TRANSFORMS
        shuffle = False
    dataset = torchvision.datasets.CIFAR10(root=args.data, train=train,
                                           download=True, transform=transform)
    
    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                       shuffle=shuffle)
