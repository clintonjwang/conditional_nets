import gc
import sys
import glob, shutil, os
from os.path import *
import pickle
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.backends.cudnn.benchmark=True

sys.path.append('..')
import niftiutils.nn.logger as logs
import niftiutils.io as io
import niftiutils.nn.submodules as subm
import networks.base as nets
import networks.dense as dense
import datasets.common as datasets
import datasets.mnist as mnist
import datasets.cifar as cifar
import util
import analysis as ana

import warnings
warnings.filterwarnings("ignore")

csv_path = '/data/vision/polina/users/clintonw/code/vision_final/results.csv'
log_path = '/data/vision/polina/users/clintonw/code/vision_final/logs'

arg_cols = ['model_type', 'N_train', 'dataset', 'nZ', 'nU', 'context_dist', 'Y_fn', 'noise_p', 'noise_lim', 'optim', 'arch', 'lr', 'wd', 'emp_est_acc', 'true_est_acc']
result_cols = ['acc', 'true_KL', 'emp_KL', 'true_JS', 'emp_JS', 'epochs', 'timestamp']

def write_df(args):
    if exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
        if arg_cols + result_cols != list(df.columns):
            if len(df) == 0:
                df = pd.DataFrame(columns=arg_cols + result_cols)
            else:
                raise ValueError('Fix columns in results.csv or delete it.')
    else:
        df = pd.DataFrame(columns=arg_cols + result_cols)

    ix = 0
    args['model_name'] = args['model_type'] + '_%d' % ix
    while args['model_name'] in df.index:
        ix += 1
        args['model_name'] = args['model_type'] + '_%d' % ix

    return args

    df.loc[args['model_name']] = [args[k] for k in arg_cols] + [-1]*(len(result_cols)-1) + [time.time()]
    df.to_csv(csv_path)


def main(args):
    n_gpus = torch.cuda.device_count()
    bsz = args['bsz'] * n_gpus
    n_samples = 100

    if 'mnist' in args['dataset']:
        dims = (1,28,28)
    else:
        dims = (3,32,32)

    if 'fmnist' == args['dataset']:
        dataset = mnist.FMnistDS
    elif 'mnist' == args['dataset']:
        dataset = mnist.MnistDS
    elif 'svhn' == args['dataset']:
        dataset = cifar.SVHN
    elif 'cifar10' == args['dataset']:
        dataset = cifar.Cifar10
    elif 'cifar100' == args['dataset']:
        dataset = cifar.Cifar100
        
    ds = dataset(train=True, args=args)
    train_loader = datasets.get_loader(ds, bsz=bsz)
    assert args['N_train'] == len(ds), "Training set only has %d elements, N_train was specified as %d" % (len(ds), args['N_train'])
    train_zuy = ds.synth_vars
    
    ds = dataset(train=False, args=args)
    val_loader = datasets.get_loader(ds, bsz=bsz)
    N_val = len(ds)
    
    if args['img_only']:
        if args['arch'] == 'all-conv':
            model = nets.BaseCNN(n_cls=args['nZ'], dims=dims).cuda()
        elif args['arch'] == 'dense':
            model = dense.densenet(depth=64, k=16, num_classes=args['nZ']).cuda()
        elif args['arch'] == 'sota-dense':
            model = dense.densenet(depth=250, k=24, num_classes=args['nZ']).cuda()
    else:
        model = nets.FilmCNN(args, dims=dims).cuda()
        
        entropy = ana.get_entropy(args=args)
        args = {**args, **entropy}
        # accuracies of the estimators with access to the image labels
        args['emp_est_acc'], args['true_est_acc'] = ana.get_accs(train_zuy=train_zuy, val_zuy=ds.synth_vars, args=args)

    args = write_df(args)

    par_model = nn.DataParallel(model)
    
    if args['tboard']:
        logger = logs.Logger(log_path)

    if args['optim'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=args['wd'])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150, 200], gamma=0.1)
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 225, 300], gamma=0.1)
    elif args['optim'] == 'nest':
        optimizer = optim.SGD(model.parameters(), momentum=.9, nesterov=True, lr=0.1, weight_decay=args['wd'])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150, 200], gamma=0.1)
    elif args['optim'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
        
    criterion = nn.CrossEntropyLoss().cuda()
    max_epochs = args['epochs']
    epoch = 1
    patience = args['patience']
    loss_hist = [np.inf]*patience
    hist = {'loss': [], 'val-acc': []}

    while epoch <= max_epochs:
        if 'scheduler' in locals():
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
                context = torch.zeros(labels.size(0), args['nU'], dtype=torch.float).cuda()
                context.scatter_(1, labels[:,1].view(-1,1), 1.)
                target = labels[:,-1].long()
                pred = par_model(imgs, context)
            
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        running_loss /= args['N_train']
        
        if running_loss < np.min(loss_hist):
            torch.save(model.state_dict(), "../results/%s.state" % args['model_name'])
            
        gc.collect()
        torch.cuda.empty_cache()
        par_model.eval();

        print("Loss: %.2f / " % (running_loss), end='')
        if imgs.size(1) == 1:
            sample_train_imgs = imgs[:5,0].cpu().numpy()
        else:
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
                context = torch.zeros(labels.size(0), args['nU'], dtype=torch.float).cuda()
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

    hist['acc'] = hist['val-acc'][-1]
    hist['epochs'] = epoch-1
    hist['timestamp'] = time.time()
    
    if not args['img_only']:
        hist['true_KL'], hist['emp_KL'], hist['true_JS'], hist['emp_JS'] = ana.get_stats(ds.synth_vars, args, preds)
        hist = {**hist, **args}
        hist.pop('f');
        
    io.pickle_dump(hist, "../results/%s.hist" % args['model_name'])
        
    df = pd.read_csv(csv_path, index_col=0)
    df.loc[args['model_name']] = [hist[k] if k in hist else -1 for k in arg_cols + result_cols]
    df.to_csv(csv_path)
        

def run_model(M, loader, args, optimizer=None):
    train = optimizer is not None
    total_loss,acc,den = 0.,0.,0.001
    for batch in loader:
        if args['modality'] is 'none':
            clinvars, accnums = batch
            mRS = clinvars[:,0].cuda()
            clinvars = clinvars[:,1:].cuda()
            #accnums = np.array(accnums)
        else:
            imgs, clinvars, accnums = batch
            if args['modality'] == 'all':
                imgs, clinvars, mRS, accnums = mutil.split_gpus(imgs, clinvars, accnums)
            else:
                imgs = imgs.cuda()
                mRS = clinvars[:,0].cuda()
                clinvars = clinvars[:,1:].cuda()
                #accnums = np.array(accnums)

        if train:
            optimizer.zero_grad()
        
        pred = M(imgs, clinvars)
        
        if args['modality'] == 'all':
            clinvars = torch.cat(clinvars, dim=0)
        losses = [torch.abs(pred[:,C.maxmRS+C.xent:][clinvars != 0] - clinvars[clinvars != 0]).mean()]
        if sum(mRS >= 0) > 0:
            pred = pred[mRS >= 0, :C.maxmRS+C.xent]
            mRS = mRS[mRS >= 0].long()
            losses.append(args['loss_fn'](pred, mRS) * 50)

            if not C.xent:
                pred = nn_loss.get_marginals(pred)

            pred = torch.max(pred, dim=1)[1]

            acc += (pred == mRS).sum().item()
            den += len(mRS)
        if torch.isnan(losses[0]):
            losses = losses[1:]
            if len(losses) == 0:
                continue
        loss = sum(losses)
        
        if train:
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item()

    return total_loss, acc/den


def get_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='fmnist', choices=['mnist', 'fmnist', 'svhn', 'cifar10', 'cifar100'], help='mnist, fmnist, svhn, cifar10 or cifar100')
    parser.add_argument('--N_train', type=int, default=50000, help='number of training examples')
    parser.add_argument('--nU', type=int, default=10, help='Number of possible categories for the context variable.')
    parser.add_argument('--context_dist', type=str, default='uniform', choices=['uniform', 'binomial'], help='Distribution of the context variable.')
    parser.add_argument('--noise_p', type=float, default=0.1, help='Amount of probability to move away from the mode for the noise (+0).')
    parser.add_argument('--noise_lim', type=int, default=2, help='Maximum displacement by noise.')
    parser.add_argument('--Y_fn', type=str, default='1+1d1', help='Outcome as a function of z and u. "A+BdC" for (z*A+u*B)//C')
    
    parser.add_argument('--h_dim', type=int, default=32, help='Number of latent dimensions (does not necessarily match the true number of classes).')
    parser.add_argument('--arch', type=str, default='all-conv', choices=['all-conv', 'dense', 'sota-dense'], help='CNN architecture.')
    parser.add_argument('--u_arch', type=str, default='film', choices=['film', 'cat', 'gan'], help='How the contextual variables are incorporated into the network.')
    parser.add_argument('--optim', type=str, default='nest', choices=['sgd', 'nest', 'adam'], help='Optimizer.')
    parser.add_argument('--patience', type=int, default=15, help='Loss patience.')
    parser.add_argument('--epochs', type=int, default=500, help='Loss patience.')

    parser.add_argument('--mcmc', action="store_true")
    parser.add_argument('--sgld', action="store_true")
    parser.add_argument('--tboard', action="store_true")
    parser.add_argument('--refresh_data', action="store_true")
    parser.add_argument('--img_only', action="store_true")

    parser.add_argument('--bsz', type=int, default=2048, help='batch size per GPU')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')

    if args is not None:
        args = vars(parser.parse_args(args))
    else:
        args = vars(parser.parse_args())

    args['nZ'] = 100 if 'cifar100' == args['dataset'] else 10
    if args['arch'] == 'sota-dense':
        args['bsz'] = 16
    elif args['arch'] == 'dense':
        args['bsz'] = 256
        
    i = args['Y_fn'].find('+')
    j = args['Y_fn'].find('d')
    A = int(args['Y_fn'][:i])
    B = int(args['Y_fn'][i+1:j])
    C = int(args['Y_fn'][j+1:])
    args['f'] = lambda z,u: (z*A+u*B)//C

    if args['img_only']:
        args['model_type'] = args['dataset']
        args['nU'] = 0
    else:
        args['nY'] = args['f'](args['nZ']-1, args['nU']-1) + (args['noise_p'] > 0)*args['noise_lim'] + 1
        args['model_type'] = 'N%d%s_u%d%s_y%s_n%.2f' % (args['N_train'], args['dataset'], args['nU'], args['context_dist'], args['Y_fn'], args['noise_p'])
        
    if args['noise_p'] == 0:
        args['noise_lim'] = 0

    return args


if __name__ == '__main__':
    main(get_args())
