import gc
import sys
import glob, shutil, os
from os.path import *
import pickle
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.backends.cudnn.benchmark=True

sys.path.append('..')
import niftiutils.helper_fxns as hf
import niftiutils.nn.submodules as subm
import networks.base as nets
import datasets.common as datasets
import datasets.mnist as mnist
import datasets.cifar as cifar
import util

csv_path = '/data/vision/polina/users/clintonw/code/vision_final/results.csv'

def main(opts):
    if opts['img_only']:
        opts['model_name'] = opts['dataset']
    else:
        opts['model_name'] = '%s_u%d%s_y%s_n%d' % (opts['dataset'], opts['nU'], opts['context_dist'], opts['outcome_fn'], opts['noise'])
        
    if opts['outcome_fn'] == '+':
        f = lambda h,u: h+u
    elif opts['outcome_fn'] == '+/3':
        f = lambda h,u: (h+u)//3
    elif opts['outcome_fn'] == '+/5':
        f = lambda h,u: (h+u)//5
    elif opts['outcome_fn'] == '+/10':
        f = lambda h,u: (h+u)//10
    opts['f'] = f
        
    arg_cols = ['dataset', 'nU', 'context_dist', 'noise']
    if exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
    else:
        df = pd.DataFrame(columns=arg_cols + ['acc', 'true_KL', 'emp_KL'])
    
    n_gpus = torch.cuda.device_count()
    bsz = 1024*n_gpus
    n_samples = 100

    n_cls = 10
    dims = (1,28,28)
    if 'fmnist' == opts['dataset']:
        dataset = mnist.FMnistDS
    elif 'mnist' == opts['dataset']:
        dataset = mnist.MnistDS
    elif 'svhn' == opts['dataset']:
        dataset = 'svhn'
    elif 'cifar10' == opts['dataset']:
        dataset = cifar.Cifar10
        dims = (3,32,32)
    elif 'cifar100' == opts['dataset']:
        n_cls = 100
        dataset = cifar.Cifar100
        dims = (3,32,32)
        
    ds = dataset(train=True, args=opts)
    train_loader = datasets.get_loader(ds, bsz=bsz)
    N_train = len(ds)
    
    ds = dataset(train=False, args=opts)
    val_loader = datasets.get_loader(ds, bsz=bsz)
    N_val = len(ds)

    #train_loader, val_loader = dl.get_split_loaders(ds, batch_size)
    
    if opts['img_only']:
        model = nets.BaseCNN(n_cls=n_cls, dims=dims).cuda()
    else:
        model = nets.FilmCNN(n_cls=n_cls, dims=dims).cuda()
    par_model = nn.DataParallel(model)
    
    optimizer = optim.Adam(model.parameters(), lr=opts['lr'])
    criterion = nn.CrossEntropyLoss(ignore_index=-1).cuda()
    max_epochs = 200
    epoch = 1
    patience = 5
    loss_hist = [np.inf]*patience
    hist = {'loss': [], 'val-acc': []}

    while epoch <= max_epochs:
        running_loss = 0.0
        print('Epoch: ' + str(epoch))
        par_model.train();

        for batch_num, (imgs, labels) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            
            imgs, labels = imgs.cuda(), labels.cuda()
            if 'img' in opts['model_name']:
                target = labels[:,0].long()
                if opts['mcmc']:
                    pred = torch.stack([par_model(imgs) for _ in range(n_samples)], 0).mean(0)
                else:
                    pred = par_model(imgs)
            else:
                context = torch.zeros(labels.size(0), 10, dtype=torch.float).cuda()
                context.scatter_(1, labels[:,1].view(-1,1), 1.)
                target = labels[:,-1].long()
                pred = par_model(imgs, context)
            
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if running_loss < np.min(loss_hist):
            torch.save(model.state_dict(), "../history/%s.state" % opts['model_name'])
            
        gc.collect()
        torch.cuda.empty_cache()
        par_model.eval();

        print("Loss: %.2f" % (running_loss))
        
        acc = 0.
        preds = []
        for batch_num, (imgs, labels) in enumerate(val_loader, 1):
            imgs, labels = imgs.cuda(), labels.cuda()
            if opts['img_only']:
                target = labels[:,0].long()
                if opts['mcmc']:
                    post = torch.stack([par_model(imgs) for _ in range(n_samples)], 0)
                    pred = post.mean(0)
                else:
                    pred = par_model(imgs)
            else:
                target = labels[:,-1].long()
                context = torch.zeros(labels.size(0), 10, dtype=torch.float).cuda()
                context.scatter_(1, labels[:,1].view(-1,1), 1.)
                pred = par_model(imgs, context)
            
            est = torch.max(pred, dim=1)[1]
            if opts['mcmc']:
                mode_acc += (est == post).sum().item()
            acc += (est == target).sum().item()
            
            pred = F.softmax(pred, dim=1)
            preds.append(pred.detach().cpu().numpy())
            
        preds = np.concatenate(preds,0)
        
        acc /= N_val
        if 'mcmc' in opts['model_name']:
            print("Validation accuracy: mean estimator %.1f%%, mode estimator %.1f%%" % (100*acc, 100*mode_acc))
        else:
            print("Validation accuracy: %.1f%%" % (100*acc))
        hist['loss'].append(running_loss)
        hist['val-acc'].append(acc)
            
        loss_hist.append(running_loss)
        if np.min(loss_hist[patience:]) >= loss_hist[-1-patience]:
            break
            
        gc.collect()
        epoch += 1

    hf.pickle_dump(hist, "../history/%s.history" % opts['model_name'])
    if not opts['img_only']:
        xuy = ds.synth_vars
        emp_post = util.emp_post(xuy)
        true_post = util.true_post(xuy[:,:2], f, noise=opts['noise'])
        pred_post, pX, pU = util.pred_post(preds, xuy[:,:2])
        true_KL = util.kl_div(pred_post, true_post, pX, pU)
        emp_KL = util.kl_div(pred_post, emp_post, pX, pU)
    else:
        true_KL, emp_KL = -1,-1
    df.loc[model] = [opts[k] for k in arg_cols] + [acc, true_KL, emp_KL]
    df.to_csv(csv_path)
        

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fmnist', 'svhn', 'cifar10', 'cifar100'], help='mnist, fmnist, svhn, cifar10 or cifar100')
    parser.add_argument('--nU', type=int, default=10, help='Number of possible categories for the context variable.')
    parser.add_argument('--context_dist', type=str, default='uniform', choices=['uniform', 'binomial'], help='Distribution of the context variable.')
    parser.add_argument('--noise', type=int, default=0, help='Noise in the outcome variable.')
    parser.add_argument('--outcome_fn', type=str, default='+', choices=['+', '+/3', '+/5', '+/10'], help='Outcome as a function of h and u.')
    parser.add_argument('--mcmc', type=bool, default=False)
    parser.add_argument('--refresh_data', type=bool, default=False)
    parser.add_argument('--img_only', type=bool, default=False)

    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')

    return parser.parse_args()


if __name__ == '__main__':
    main(vars(get_args()))
