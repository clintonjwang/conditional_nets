import os
from os.path import *
import numpy as np
import scipy.stats as stats
import torch
import torch.utils.data as data
from torchvision.transforms import *
import torchvision.datasets as torch_data

import sys
sys.path.append('..')
import datasets.common

root_dir = '/data/vision/polina/users/clintonw/code/vision_final'

class SVHN(torch_data.SVHN):
    train_mean = [0.43768210460907236,  0.44376970456396997,  0.4728044222297246]
    train_std = [0.19803012447157164,  0.20101562471828893,  0.19703614172172376]
    transform_train = Compose([
        ToPILImage(),
        RandomCrop(32, padding=4),
        ToTensor(),
        Normalize(train_mean, train_std),
    ])
    transform_test = Compose([
        ToPILImage(),
        ToTensor(),
        Normalize(train_mean, train_std),
    ])

    def __init__(self, train, args, root=root_dir + '/data/svhn', **kwargs):
        super(SVHN, self).__init__(root=root, split='train' if train else 'test', **kwargs)
        if train:
            if 'ae' not in args['arch']:
                self.data = self.data[:args['N_train']]
            self.imgs = np.transpose(self.data, (0,2,3,1))
            self.labels = self.labels[:args['N_train']]
            fn = join(self.root, '%s_train.npy' % args['model_type'])
        else:
            self.imgs = np.transpose(self.data, (0,2,3,1))
            fn = join(self.root, '%s_val.npy' % args['model_type'])
        
        if args['refresh_data'] or not exists(fn):
            datasets.common.get_cls_data(self, fn=fn, args=args)
        
        self.synth_vars = torch.from_numpy(np.load(fn))
        self.train = train

    def __getitem__(self, index):
        img, synth_vars = self.imgs[index], self.synth_vars[index]
        if self.train:
            img = self.transform_train(img)
        else:
            img = self.transform_test(img)
        
        return img, synth_vars
    
    def __len__(self):
        return len(self.labels)
        

class Cifar10(torch_data.CIFAR10):
    train_mean = [0.49139968,  0.48215841,  0.44653091]
    train_std = [0.24703223,  0.24348513,  0.26158784]
    transform_train = Compose([
        ToPILImage(),
        RandomCrop(32, padding=4),#ColorJitter(brightness=0.1, contrast=0.2, saturation=0.02, hue=0.02), RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(train_mean, train_std),
    ])
    transform_test = Compose([
        ToPILImage(),
        ToTensor(),
        Normalize(train_mean, train_std),
    ])

    def __init__(self, train, args, root=root_dir + '/data/cifar10', **kwargs):
        super(Cifar10, self).__init__(root=root, train=train, **kwargs)
        if train:
            self.imgs = self.train_data[:args['N_train']]
            self.labels = self.train_labels[:args['N_train']]
            fn = join(self.root, '%s_train.npy' % args['model_type'])
        else:
            self.imgs = self.test_data
            self.labels = self.test_labels
            fn = join(self.root, '%s_val.npy' % args['model_type'])

        if args['refresh_data'] or not exists(fn):
            datasets.common.get_cls_data(self, fn=fn, args=args)
        
        self.synth_vars = torch.from_numpy(np.load(fn))
        self.train = train

    def __getitem__(self, index):
        img, synth_vars = self.imgs[index], self.synth_vars[index]
        if self.train:
            img = self.transform_train(img)
        else:
            img = self.transform_test(img)
        
        return img, synth_vars
        
    def __len__(self):
        return len(self.labels)

class Cifar100(Cifar10):
    train_mean = [0.507, 0.487, 0.441]
    train_std = [0.267, 0.256, 0.276]
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

    def __init__(self, train, args, **kwargs):
        super(Cifar100, self).__init__(root=root_dir + '/data/cifar100', train=train, args=args, **kwargs)
    