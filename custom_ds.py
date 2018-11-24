import os
from os.path import *
import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets

class MnistDS(datasets.MNIST):
    def __init__(self, train, mode='baseline', noise=.1, refresh_data=False, root='data/mnist', **kwargs):
        super(MnistDS, self).__init__(root=root, train=train, **kwargs)
        self.mode = mode
        assert self.mode in ['baseline', 'correlated', 'missing', 'context-no-info', 'img-no-info', 'multiple']
        
        self.noise = noise
        if train:
            self.data = self.train_data
            self.labels = self.train_labels
        else:
            self.data = self.test_data
            self.labels = self.test_labels

        if refresh_data or not exists(join(self.root, 'train_%s.bin' % self.mode)):
            self.refresh_data()
            
        if train:
            self.synth_vars = np.load(join(self.root, 'train_%s.npy' % self.mode))
        else:
            self.synth_vars = np.load(join(self.root, 'test_%s.npy' % self.mode))
        self.train = train

    def __getitem__(self, index):
        img, target = self.imgs[index], self.synth_vars[index]
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def refresh_data(self, regression=False):
        h = self.labels.numpy()
        if self.noise:
            noise = np.random.normal(0,self.noise,h.shape)
        else:
            noise = 0
            
        # generate the contextual variable u
        if self.mode == 'correlated':
            u = np.random.binomial(9,(h+1)/11)
        elif self.mode == 'missing': #only interesting in regression, not classification
            u = np.random.binomial(9,(h+1)/11)
            hide = np.random.binomial(1,(h-u+1)/(h+3))
        else:
            u = np.random.randint(0,10,h.shape)

        # generate the outcome y
        if self.mode == 'context-no-info':
            y = np.round(np.clip(h*2, 0,18))
            y_true = h*2
        elif self.mode == 'img-no-info':
            y = np.round(np.clip(u*2, 0,18))
            y_true = u*2
        elif self.mode == 'multiple':
            v = np.random.binomial(6,(h+1)/11)
            w = np.random.binomial(6,(10-u)/11)
            y = np.round(np.clip(h + u - v + w + np.random.normal(0,self.noise,h.shape), 0,18)/3)
            #v = np.random.chisquare(4,h.shape) #self.model as gaussian
            #v -= v.mean()
            #w = np.random.binomial(5,(h+1)/11) #self.model as categorical
            #y = np.round(np.clip(h + u - v + w*(w==3+w==4) + np.random.normal(0,self.noise,h.shape), 0,18)/3)
        else:
            y = np.round(np.clip(h + u + noise, 0, 18))
            y_true = h + u

        if self.mode == 'missing':
            u[hide == 1] = -1
        
        if self.mode == 'multiple':
            synth_vars = np.stack([h, u, v, w, y, y_true], 1)
        else:
            synth_vars = np.stack([h, u, y, y_true], 1)

        if self.train:
            np.save(join(self.root, 'train_%s.npy' % self.mode), synth_vars)
        else:
            np.save(join(self.root, 'test_%s.npy' % self.mode), synth_vars)
        
        
class FMnistDS(MnistDS):
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]
    def __init__(self, train, **kwargs):
        super(FMnistDS, self).__init__(root='data/fmnist', train=train, **kwargs)
    