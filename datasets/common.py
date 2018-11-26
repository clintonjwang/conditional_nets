import os
from os.path import *
import numpy as np
import scipy.stats as stats
import torch
import torch.utils.data as data
from torchvision import datasets, transforms

def get_loader(ds, bsz, random_seed=42):
    np.random.seed(random_seed)
    kwargs = {'batch_size': bsz, 'num_workers': 4, 'pin_memory': True}
    dl = data.DataLoader(ds, **kwargs)

    return dl

def get_split_loaders(ds, batch_size, random_seed=42):
    np.random.seed(random_seed)
    ixs = range(len(ds))
    train_sz = int(np.round(.7*len(ixs)))
    np.random.shuffle(ixs)
    ds.train_indices = ixs[:train_sz]
    val_indices = ixs[train_sz:]

    kwargs = {'batch_size': batch_size, 'num_workers': 6, 'pin_memory': True}
    train_sampler = SubsetRandomSampler(ds.train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = data.DataLoader(ds, sampler=train_sampler, **kwargs)
    val_loader = data.DataLoader(ds, sampler=val_sampler, **kwargs)

    return train_loader, val_loader


def get_cls_data(ds, f, noise, mode, fn):
    h = ds.labels.numpy()
    # generate the contextual variable u
    if mode == 'correlated':
        u = np.random.binomial(9,(h+1)/11)
    elif mode == 'high-dim':
        u = np.random.randint(0,10,(h.shape[0],64))
    else:
        u = np.random.randint(0,10,h.shape)

    # generate the outcome y
    if mode == 'high-dim':
        y = np.round(u.sum(1)/64) + h
    else:
        y = f(h,u) + np.random.randint(low=0,high=noise+1, size=h.shape)

    if mode == 'high-dim':
        synth_vars = np.concat([h.expand_dims(1), u, y.expand_dims(1)], 1)
    else:
        synth_vars = np.stack([h, u, y], 1)

    np.save(fn, synth_vars)