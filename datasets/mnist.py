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

class MnistDS(torch_data.MNIST):
    def __init__(self, train, args, root=root_dir + '/data/mnist', **kwargs):
        super(MnistDS, self).__init__(root=root, train=train, **kwargs)
        if train:
            self.imgs = self.train_data
            self.labels = self.train_labels
            fn = join(self.root, '%s_train.npy' % args['model_type'])
        else:
            self.imgs = self.test_data
            self.labels = self.test_labels
            fn = join(self.root, '%s_val.npy' % args['model_type'])
        
        if args['refresh_data'] or not exists(fn):
            datasets.common.get_cls_data(self, fn=fn, args=args)
        
        #self.imgs = self.imgs.view(-1,1,28,28).numpy()
        self.imgs = self.imgs.view(-1,1,28,28).float() / 255.
        
        self.synth_vars = torch.from_numpy(np.load(fn))
        self.train = train

    def __getitem__(self, index):
        img, synth_vars = self.imgs[index], self.synth_vars[index]
        
        return img, synth_vars
    
        
class FMnistDS(MnistDS):
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]
    def __init__(self, train, **kwargs):
        super(FMnistDS, self).__init__(root=root_dir + '/data/fmnist', train=train, **kwargs)
    