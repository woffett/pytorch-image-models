'''
An addition to the timm library that includes compatibility with CIFAR10
and its standard augmentations

Adapted from:
https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
'''

import torch
import torchvision
import torchvision.transforms as transforms
from .transforms import transforms_cifar10_train, transforms_cifar10_eval

def cifar10_loader(args, train=False):
    if train:
        transform = transforms_cifar10_train()
        shuffle = True
    else:
        transform = transforms_cifar10_eval()
        shuffle = False
    dataset = torchvision.datasets.CIFAR10(root=args.data, train=train,
                                           download=True, transform=transform)
    
    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                       shuffle=shuffle)
