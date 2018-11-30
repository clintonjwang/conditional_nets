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

def get_split_loaders(ds, batch_size, N_train, random_seed=42):
    np.random.seed(random_seed)
    ixs = range(len(ds))
    train_sz = N_train #int(np.round(.7*len(ixs)))
    np.random.shuffle(ixs)
    ds.train_indices = ixs[:train_sz]
    val_indices = ixs[train_sz:]

    kwargs = {'batch_size': batch_size, 'num_workers': 8, 'pin_memory': True}
    train_sampler = SubsetRandomSampler(ds.train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = data.DataLoader(ds, sampler=train_sampler, shuffle=True, **kwargs)
    val_loader = data.DataLoader(ds, sampler=val_sampler, **kwargs)

    return train_loader, val_loader


def get_cls_data(ds, args, fn):
    z = ds.labels.numpy()
    if args['img_only']:
        np.save(fn, np.expand_dims(z, 1))
        return
    nU = args['nU']
    
    # generate the contextual variable u
    if args['context_dist'] == 'binomial':
        u = np.random.binomial(nU, (z+1)/(args['nZ']+1))
    elif args['context_dist'] == 'multinomial':
        u = np.random.randint(0,nU, (z.shape[0],64))
    else:
        u = np.random.randint(0,nU, z.shape)

    # generate the outcome y
    if args['context_dist'] == 'multinomial':
        y = np.round(u.sum(1)/64) + z
    elif args['noise']:
        y = args['f'](z,u) + np.random.binomial(n=args['noise'], p=2/3, size=z.shape)
    else:
        y = args['f'](z,u)

    if args['context_dist'] == 'multinomial':
        zuy = np.concat([z.expand_dims(1), u, y.expand_dims(1)], 1)
    else:
        zuy = np.stack([z, u, y], 1)

    np.save(fn, zuy)