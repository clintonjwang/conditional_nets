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
import niftiutils.nn.logger as logs
import niftiutils.helper_fxns as hf
import niftiutils.nn.submodules as subm
import networks.base as nets
import datasets.common as datasets
import datasets.mnist as mnist
import datasets.cifar as cifar
import util

csv_path = '/data/vision/polina/users/clintonw/code/vision_final/results.csv'
log_path = '/data/vision/polina/users/clintonw/code/vision_final/logs'

def main(args):
    arg_cols = ['model_type', 'dataset', 'nU', 'context_dist', 'noise', 'optim', 'lr', 'wd']
    if exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
        if len(set(arg_cols).difference(df.columns)) > 0:
            df = pd.DataFrame(columns=arg_cols + ['acc', 'true_KL', 'emp_KL'])
    else:
        df = pd.DataFrame(columns=arg_cols + ['acc', 'true_KL', 'emp_KL'])

    args['model_name'] = args['model_type'] + '_%d' % sum(df['model_type'] == args['model_type'])
    df.loc[args['model_name']] = [args[k] for k in arg_cols] + [-1,-1,-1]
    df.to_csv(csv_path)

    n_gpus = torch.cuda.device_count()
    bsz = args['bsz'] * n_gpus
    n_samples = 100

    n_cls = 10
    dims = (1,28,28)
    if 'fmnist' == args['dataset']:
        dataset = mnist.FMnistDS
    elif 'mnist' == args['dataset']:
        dataset = mnist.MnistDS
    elif 'svhn' == args['dataset']:
        dataset = 'svhn'
    elif 'cifar10' == args['dataset']:
        dataset = cifar.Cifar10
        dims = (3,32,32)
    elif 'cifar100' == args['dataset']:
        n_cls = 100
        dataset = cifar.Cifar100
        dims = (3,32,32)
        
    ds = dataset(train=True, args=args)
    train_loader = datasets.get_loader(ds, bsz=bsz)
    N_train = len(ds)
    
    ds = dataset(train=False, args=args)
    val_loader = datasets.get_loader(ds, bsz=bsz)
    N_val = len(ds)

    #train_loader, val_loader = dl.get_split_loaders(ds, batch_size)
    n_out = args['f'](n_cls-1, args['nU']-1) + args['noise'] + 1
    
    if args['img_only']:
        model = nets.BaseCNN(n_cls=n_cls, dims=dims).cuda()
    else:
        model = nets.FilmCNN(n_cls=n_cls, dims=dims, n_context=args['nU'], n_out=n_out).cuda()
    par_model = nn.DataParallel(model)
    
    if args['tboard']:
        logger = logs.Logger(log_path)

    if args['optim'] == 'nest':
        optimizer = optim.SGD(model.parameters(), momentum=.9, nesterov=True, lr=args['lr'], weight_decay=args['wd'])
    elif args['optim'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 250, 300], gamma=0.1)

    criterion = nn.CrossEntropyLoss(ignore_index=-1).cuda()
    max_epochs = 2000
    epoch = 1
    patience = 20
    loss_hist = [np.inf]*patience
    hist = {'loss': [], 'val-acc': []}

    while epoch <= max_epochs:
        if args['optim'] == 'sched':
            scheduler.step()

        running_loss = 0.0
        print('Epoch: %d / ' % (epoch), end='')
        par_model.train();

        for imgs, labels in train_loader:
            optimizer.zero_grad()
            
            imgs, labels = imgs.cuda(), labels.cuda()
            if args['img_only']:
                target = labels[:,0].long()
                if args['mcmc']:
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
            torch.save(model.state_dict(), "../history/%s.state" % args['model_name'])
            
        gc.collect()
        torch.cuda.empty_cache()
        par_model.eval();

        print("Loss: %.2f / " % (running_loss), end='')
        sample_train_imgs = imgs[:5].cpu().numpy()
        
        acc = 0.
        preds = []
        for imgs, labels in val_loader:
            imgs, labels = imgs.cuda(), labels.cuda()
            if args['img_only']:
                target = labels[:,0].long()
                if args['mcmc']:
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
            if args['mcmc']:
                mode_acc += (est == post).sum().item()
            acc += (est == target).sum().item()
            
            pred = F.softmax(pred, dim=1)
            preds.append(pred.detach().cpu().numpy())
            
        preds = np.concatenate(preds,0)
        
        acc /= N_val
        if args['mcmc']:
            print("Val accuracy: mean estimator %.1f%%, mode estimator %.1f%%" % (100*acc, 100*mode_acc))
        else:
            print("Val accuracy: %.1f%%" % (100*acc))
        hist['loss'].append(running_loss)
        hist['val-acc'].append(acc)
            
        loss_hist.append(running_loss)
        if np.min(loss_hist[patience:]) >= loss_hist[-1-patience]:
            break
            
        gc.collect()
        epoch += 1

        if args['tboard']:
            info = { 'loss': running_loss, 'accuracy': acc }
            logs.log_tboard(logger, info, model, sample_train_imgs, epoch)

    hf.pickle_dump(hist, "../history/%s.history" % args['model_name'])
    if not args['img_only']:
        xuy = ds.synth_vars
        emp_post = util.emp_post(xuy)
        true_post = util.true_post(xuy[:,:2], args['f'], noise=args['noise'])
        pred_post, pX, pU = util.pred_post(preds, xuy[:,:2])
        true_KL = util.kl_div(pred_post, true_post, pX, pU)
        emp_KL = util.kl_div(pred_post, emp_post, pX, pU)
    else:
        true_KL, emp_KL = -1,-1
        
    df.loc[args['model_name']] = [args[k] for k in arg_cols] + [acc, true_KL, emp_KL]
    df.to_csv(csv_path)
        

def get_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fmnist', 'svhn', 'cifar10', 'cifar100'], help='mnist, fmnist, svhn, cifar10 or cifar100')
    parser.add_argument('--nU', type=int, default=10, help='Number of possible categories for the context variable.')
    parser.add_argument('--context_dist', type=str, default='uniform', choices=['uniform', 'binomial'], help='Distribution of the context variable.')
    parser.add_argument('--noise', type=int, default=0, help='Noise in the outcome variable.')
    parser.add_argument('--outcome_fn', type=str, default='+', choices=['+', '+3', '+5', '+10'], help='Outcome as a function of h and u.')
    parser.add_argument('--optim', type=str, default='adam', choices=['nest', 'adam'], help='Optimizer.')

    parser.add_argument('--mcmc', action="store_true")
    parser.add_argument('--tboard', action="store_true")
    parser.add_argument('--refresh_data', action="store_true")
    parser.add_argument('--img_only', action="store_true")

    parser.add_argument('--bsz', type=int, default=1024, help='batch size per GPU')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.001, help='weight decay')

    if args is not None:
        args = vars(parser.parse_args(args))
    else:
        args = vars(parser.parse_args())

    if args['img_only']:
        args['model_type'] = args['dataset']
    else:
        args['model_type'] = '%s_u%d%s_y%s_n%d' % (args['dataset'], args['nU'], args['context_dist'], args['outcome_fn'], args['noise'])
        
    if args['outcome_fn'] == '+':
        f = lambda h,u: h+u
    elif args['outcome_fn'] == '+3':
        f = lambda h,u: (h+u)//3
    elif args['outcome_fn'] == '+5':
        f = lambda h,u: (h+u)//5
    elif args['outcome_fn'] == '+10':
        f = lambda h,u: (h+u)//10
    args['f'] = f

    return args


if __name__ == '__main__':
    main(get_args())
