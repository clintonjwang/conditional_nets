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

class Cifar10(torch_data.CIFAR10):
    def __init__(self, train, args, root=root_dir + '/data/cifar10', **kwargs):
        super(Cifar10, self).__init__(root=root, train=train, **kwargs)
        if train:
            self.imgs = self.train_data
            self.labels = torch.tensor(self.train_labels, dtype=torch.long)
            fn = join(self.root, 'train_%s.npy' % args['model_name'])
        else:
            self.imgs = self.test_data
            self.labels = torch.tensor(self.test_labels, dtype=torch.long)
            fn = join(self.root, 'test_%s.npy' % args['model_name'])

        if args['refresh_data'] or not exists(fn):
            self.get_cls_data(f=args['f'], noise=args['noise'], mode=args['context_dist'], fn=fn)
        
        self.imgs = torch.tensor(self.imgs, dtype=torch.float).permute(0,3,1,2) / 255.
        
        self.synth_vars = torch.from_numpy(np.load(fn))
        self.train = train

    def __getitem__(self, index):
        img, synth_vars = self.imgs[index], self.synth_vars[index]
        
        return img, synth_vars

    def get_cls_data(self, **kwargs):
        datasets.common.get_cls_data(self, **kwargs)
        
class Cifar100(Cifar10):
    def __init__(self, train, **kwargs):
        super(Cifar100, self).__init__(root=root_dir + '/data/cifar100', train=train, **kwargs)
    