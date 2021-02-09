"""
   CIFAR-10 CIFAR-100, Tiny-ImageNet data loader
"""
from copy import deepcopy
import random
import os
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from datasets import CIFAR10, CIFAR100
from random_noise import label_noise, image_noise
from TinyImageNet import TinyImageNet


def fetch_dataloader(types, params, args):
    """
    Fetch and return train/dev dataloader with hyperparameters (params.subset_percent = 1.)
    """
    # using random crops and horizontal flip for train set
    if params.augmentation == "yes":
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.240, 0.243, 0.261))])
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.240, 0.243, 0.261))

    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.240, 0.243, 0.261))])

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.240, 0.243, 0.261))])

    if params.dataset == 'cifar10':

        if args.self_adaptive or args.noisy:
            trainset=CIFAR10(root='./data/data-cifar10', train=True, download=True, transform=train_transformer)
            devset = CIFAR10(root='./data/data-cifar10', train=False, download=True, transform=dev_transformer)
        else:
            trainset = torchvision.datasets.CIFAR10(root='./data/data-cifar10', train=True,
                                                download=True, transform=train_transformer)
            devset = torchvision.datasets.CIFAR10(root='./data/data-cifar10', train=False,
                                              download=True, transform=dev_transformer)
    elif params.dataset == 'cifar100':

        if args.self_adaptive or args.noisy:
            trainset=CIFAR100(root='./data/data-cifar100', train=True, download=True, transform=train_transformer)
            devset = CIFAR100(root='./data/data-cifar100', train=False, download=True, transform=dev_transformer)
        else:
            trainset = torchvision.datasets.CIFAR100(root='./data/data-cifar100', train=True,
                                                download=True, transform=train_transformer)
            devset = torchvision.datasets.CIFAR100(root='./data/data-cifar100', train=False,
                                              download=True, transform=dev_transformer)
    elif params.dataset == 'tiny_imagenet':

        data_dir = './data/tiny-imagenet-200/'
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])
        }

        trainset = TinyImageNet(data_dir, 'train', transform=data_transforms['train'],in_memory=False)
        devset = TinyImageNet(data_dir, 'val', transform= data_transforms['val'],in_memory=False)


    # *********************noisy train set and clean test************
    if args.noisy and args.noise_rate>0:
        noisy_train_set = deepcopy(trainset)
        print("Using noisy dataset.")
        if args.noise_type == 'corrupted_label':
            label_noise(noisy_train_set, args)
        elif args.noise_type in ['Gaussian', 'random_pixels', 'shuffled_pixels']:
            image_noise(noisy_train_set, args)
        else:
            raise ValueError("Noise type {} is not supported yet.".format(args.noise_type))
        trainset = noisy_train_set
    else:
        print("Using clean dataset.")

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
                                                      shuffle=True, num_workers=params.num_workers)

    devloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,
                                                    shuffle=False, num_workers=params.num_workers)

    if types == 'train':
        dl = trainloader
    else:
        dl = devloader
    if args.self_adaptive:
        return dl, trainset.num_classes, np.asarray(trainset.targets)
    else:
        return dl


def fetch_subset_dataloader(types, params,args):
    """
    Use only a subset of dataset for KD training, depending on params.subset_percent
    """

    # using random crops and horizontal flip for train set
    if params.augmentation == "yes":
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    if params.dataset=='cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data-cifar10', train=True,
                                                download=True, transform=train_transformer)
        devset = torchvision.datasets.CIFAR10(root='./data-cifar10', train=False,
                                              download=True, transform=dev_transformer)
    elif params.dataset=='cifar100':
        trainset = torchvision.datasets.CIFAR10(root='./data-cifar10', train=True,
                                                download=True, transform=train_transformer)
        devset = torchvision.datasets.CIFAR10(root='./data-cifar10', train=False,
                                              download=True, transform=dev_transformer)
    elif params.dataset == 'tiny_imagenet':
        data_dir = './data/tiny-imagenet-200/'
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])
        }
        train_dir = data_dir + 'train/'
        test_dir = data_dir + 'val/images/'
        trainset = torchvision.datasets.ImageFolder(train_dir, data_transforms['train'])
        devset = torchvision.datasets.ImageFolder(test_dir, data_transforms['val'])

    trainset_size = len(trainset)
    indices = list(range(trainset_size))
    split = int(np.floor(params.subset_percent * trainset_size))
    np.random.seed(230)
    np.random.shuffle(indices)

    train_sampler = SubsetRandomSampler(indices[:split])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
        sampler=train_sampler, num_workers=params.num_workers, pin_memory=params.cuda)

    devloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,
        shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)

    if types == 'train':
        dl = trainloader
    else:
        dl = devloader

    return dl