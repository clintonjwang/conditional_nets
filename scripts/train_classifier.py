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
from networks.infogan import Generator, Discriminator
import datasets.common as datasets
import datasets.mnist as mnist
import datasets.cifar as cifar
import util
import analysis as ana
import networks.dense2 as dense2

import warnings
warnings.filterwarnings("ignore")

csv_path = '/data/vision/polina/users/clintonw/code/vision_final/results.csv'
log_path = '/data/vision/polina/users/clintonw/code/vision_final/logs'

arg_cols = ['model_type', 'n_params', 'N_train', 'dataset', 'nU', 'context_dist', 'Y_fn', 'noise_p', 'noise_lim', 'optim', 'arch', 'u_arch', 'lr', 'wd', 'h_dim', 'emp_est_acc', 'true_est_acc']
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
    args['model_name'] = args['model_type'] + args['suffix'] + '_%d' % ix
    while args['model_name'] in df.index or exists("../results/%s.hist" % args['model_name']):
        ix += 1
        args['model_name'] = args['model_type'] + args['suffix'] + '_%d' % ix

    df.loc[args['model_name']] = [args[k] if k in args else -1 for k in arg_cols] + [-1]*(len(result_cols)-1) + [time.time()]
    df.to_csv(csv_path)
    
    return args


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
    
    
    if args['arch'] == 'gan':
        generator = Generator(args['h_dim'], args['cc_dim'], args['dc_dim']).cuda()
        discriminator = Discriminator(args['cc_dim'], args['dc_dim']).cuda()

        par_G = nn.DataParallel(generator)
        par_D = nn.DataParallel(discriminator)
    
        g_optimizer = optim.Adam(generator.parameters(), 0.0002, [0.5, 0.999])
        d_optimizer = optim.Adam(discriminator.parameters(), 0.001, [0.5, 0.999])

        """Train generator and discriminator."""
        fixed_noise = to_variable(torch.Tensor(np.zeros((args.sample_size, args['h_dim']))))  # For Testing
        
        epoch = 1
        args['continuous_weight'] = .5
        while epoch <= 20:
            print('Epoch: ' + str(epoch))
            par_G.train();
            par_D.train();

            g_loss_sum, d_loss_sum = 0., 0.
            for imgs, labels in train_loader:
                # ===================== Train D =====================#
                batch_size = images.size(0)
                noise = to_variable(torch.randn(batch_size, args['h_dim']))

                cc = to_variable(gen_cc(batch_size, args['cc_dim']))
                dc = to_variable(gen_dc(batch_size, args['dc_dim']))

                # Fake -> Fake & Real -> Real
                fake_images = par_G(torch.cat((noise, cc, dc),1))
                d_output_real = par_D(images)
                d_output_fake = par_D(fake_images)

                d_loss_a = -torch.mean(torch.log(d_output_real[:,0]) + torch.log(1 - d_output_fake[:,0]))

                # Mutual Information Loss
                output_cc = d_output_fake[:, 1:1+args['cc_dim']]
                output_dc = d_output_fake[:, 1+args['cc_dim']:]
                d_loss_cc = torch.mean((((output_cc - 0.0) / 0.5) ** 2))
                d_loss_dc = -(torch.mean(torch.sum(dc * output_dc, 1)) + torch.mean(torch.sum(dc * dc, 1)))

                d_loss = d_loss_a + args['continuous_weight'] * d_loss_cc + 1.0 * d_loss_dc

                # Optimization
                #discriminator.zero_grad()
                par_D.zero_grad()
                d_loss.backward(retain_graph=True)
                d_optimizer.step()

                # ===================== Train G =====================#
                # Fake -> Real
                g_loss_a = -torch.mean(torch.log(d_output_fake[:,0]))

                g_loss = g_loss_a + args['continuous_weight'] * d_loss_cc + 1.0 * d_loss_dc

                # Optimization
                #generator.zero_grad()
                par_G.zero_grad()
                g_loss.backward()
                g_optimizer.step()
                
            epoch += 1

        epoch = 1
        loss_sum = 0.
        while epoch <= num_epochs:
            print('Epoch: ' + str(epoch))
            par_G.eval();
            par_D.eval();

            running_loss = 0.
            for imgs, labels in train_loader:
                pass
                
            epoch += 1
        return
    
    if args['img_only']:
        if args['arch'] == 'all-conv':
            model = nets.BaseCNN(n_cls=args['nZ'], n_h=args['h_dim'], dims=dims).cuda()
        elif args['arch'] == 'dense':
            model = dense2.DenseNet(nClasses=args['nZ'], dims=dims, dropout=args['dropout']).cuda()
            #model = dense.densenet(depth=64, k=16, num_classes=args['nZ']).cuda()
        elif args['arch'] == 'sota-dense':
            model = dense.densenet(depth=250, k=24, num_classes=args['nZ']).cuda()
        elif args['arch'] == 'ae':
            model = nets.AE(n_cls=args['nZ'], n_h=args['h_dim'], dims=dims).cuda()
    else:
        if args['arch'] == 'all-conv':
            model = nets.FilmCNN(args, dims=dims).cuda()
        elif args['arch'] == 'dense':
            model = dense2.FilmDenseNet(args, dims=dims, dropout=args['dropout']).cuda()
        elif args['arch'] == 'ae':
            model = nets.FilmAE(args, dims=dims).cuda()
        
        entropy = ana.get_entropy(args=args)
        args = {**args, **entropy}
        # accuracies of the estimators with access to the image labels
        args['emp_est_acc'], args['true_est_acc'] = ana.get_accs(train_zuy=train_zuy, val_zuy=ds.synth_vars, args=args)

    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
    args['n_params'] = sum([np.prod(p.size()) for p in trainable_parameters])
    
    args = write_df(args)

    par_model = nn.DataParallel(model)
    
    if args['tboard']:
        logger = logs.Logger(join(log_path, args['model_name']))

    if args['optim'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=args['wd'])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 225, 300], gamma=0.1)
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.85)
    elif args['optim'] == 'nest':
        optimizer = optim.SGD(model.parameters(), momentum=.9, nesterov=True, lr=0.1, weight_decay=args['wd'])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 225, 300], gamma=0.5)
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.85)
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

        running_loss, sample_train_imgs = run_model(par_model, train_loader, args, optimizer, criterion)
        running_loss /= args['N_train']
        
        print("Loss: %.2f / " % (running_loss), end='')
        
        if running_loss < np.min(loss_hist):
            torch.save(model.state_dict(), "../results/%s.state" % args['model_name'])
            
        gc.collect()
        torch.cuda.empty_cache()
        par_model.eval();

        acc, preds = run_model(par_model, val_loader, args)
        acc /= N_val

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
    df.loc[args['model_name']] = [hist[k] if k in hist else -1 for k in df.columns]
    df.to_csv(csv_path)
        

def run_model(M, loader, args, optimizer=None, criterion=None):
    train = optimizer is not None
    acc, total_loss = 0., 0.
    preds = []
    for imgs, labels in loader:
        imgs, labels = imgs.cuda(), labels.cuda()

        if train:
            optimizer.zero_grad()

        if args['img_only']:
            target = labels[:,0].long()
            if args['mcmc']:
                post = torch.stack([M(imgs) for _ in range(n_samples)], 0)
                pred = post.mean(0)
            else:
                pred = M(imgs)
        else:
            target = labels[:,-1].long()
            context = torch.zeros(labels.size(0), args['nU'], dtype=torch.float).cuda()
            context.scatter_(1, labels[:,1].view(-1,1), 1.)
            pred = M(imgs, context)
        
        if train:
            if args['arch'] == 'ae':
                loss = criterion(pred[0], target) + .5*((pred[1] - imgs)**2).mean()
            else:
                loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        else:
            if args['arch'] == 'ae':
                pred = pred[0]
            est = torch.max(pred, dim=1)[1]
            if args['mcmc']:
                mode_acc += (est == post).sum().item()
            acc += (est == target).sum().item()
            
            pred = F.softmax(pred, dim=1)
            preds.append(pred.detach().cpu().numpy())
    
    if imgs.size(1) == 1:
        sample_train_imgs = imgs[:5,0].cpu().numpy()
    else:
        sample_train_imgs = imgs[:5].cpu().numpy()
        
    if train:
        return total_loss, sample_train_imgs
    else:
        preds = np.concatenate(preds,0)
        return acc, preds


def get_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='fmnist', choices=['mnist', 'fmnist', 'svhn', 'cifar10', 'cifar100'], help='mnist, fmnist, svhn, cifar10 or cifar100')
    parser.add_argument('--N_train', type=int, default=15000, help='number of training examples')
    parser.add_argument('--nU', type=int, default=10, help='Number of possible categories for the context variable.')
    parser.add_argument('--context_dist', type=str, default='uniform', choices=['uniform', 'binomial'], help='Distribution of the context variable.')
    parser.add_argument('--noise_p', type=float, default=0.1, help='Amount of probability to move away from the mode for the noise (+0).')
    parser.add_argument('--noise_lim', type=int, default=2, help='Maximum displacement by noise.')
    parser.add_argument('--Y_fn', type=str, default='1+1d1', help='Outcome as a function of z and u. "A+BdC" for (z*A+u*B)//C')
    
    parser.add_argument('--h_dim', type=int, default=64, help='Number of image latent dimensions (does not need to match the number of classes).')
    parser.add_argument('--arch', type=str, default='all-conv', choices=['all-conv', 'dense', 'sota-dense', 'ae', 'dense-gan', 'dense-ae'], help='CNN architecture.')
    parser.add_argument('--u_arch', type=str, default='film', choices=['film', 'cat'], help='How the contextual variables are incorporated into the network.')
    parser.add_argument('--optim', type=str, default='adam', choices=['sgd', 'nest', 'adam'], help='Optimizer.')
    parser.add_argument('--patience', type=int, default=15, help='Loss patience.')
    parser.add_argument('--epochs', type=int, default=1000, help='Loss patience.')
    parser.add_argument('--suffix', type=str, default='', help='.')

    parser.add_argument('--mcmc', action="store_true")
    parser.add_argument('--sgld', action="store_true")
    parser.add_argument('--tboard', action="store_true")
    parser.add_argument('--refresh_data', action="store_true")
    parser.add_argument('--img_only', action="store_true")

    parser.add_argument('--bsz', type=int, default=2048, help='batch size per GPU')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--dropout', type=float, default=0., help='dropout')

    parser.add_argument('--cc_dim', type=int, default=8, help='?.')
    parser.add_argument('--dc_dim', type=int, default=16, help='?.')
    
    if args is not None:
        args = vars(parser.parse_args(args))
    else:
        args = vars(parser.parse_args())

    args['nZ'] = 100 if 'cifar100' == args['dataset'] else 10
    if args['arch'] == 'sota-dense':
        args['bsz'] = 16
    elif args['arch'] == 'dense':
        args['bsz'] = 80 #256
        if args['epochs']==1000:
            args['epochs'] = 50 * np.sqrt(60000//args['N_train'])
        
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
        args['model_type'] = 'N%dK%s_u%d_y%s_n%d_%s_%s' % (args['N_train']//1000, args['dataset'], args['nU'], args['Y_fn'], args['noise_p']*100, args['arch'], args['u_arch'])
        
    if args['noise_p'] == 0:
        args['noise_lim'] = 0

    return args


if __name__ == '__main__':
    main(get_args())
