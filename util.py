import numpy as np
import pandas as pd
import scipy.stats as stats
import os, glob
from os.path import *
import scripts.train_classifier as main

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count() - 2
import importlib
importlib.reload(main)
csv_path = '/data/vision/polina/users/clintonw/code/vision_final/results.csv'

def args_to_sh(args, slurm=True, exc_gpu=False, n_gpus=4):
    ix = 0
    while exists("/data/vision/polina/users/clintonw/code/vision_final/scripts/cls%d.out" % ix):
        ix += 1

    if slurm:
        return 'nohup ./srunner.sh cls %d ' % n_gpus + ' '.join(args) + ' > srun%d.out 2> srun%d.err < /dev/null &' % (ix,ix)

    if exc_gpu == 1:
        extra = 'CUDA_VISIBLE_DEVICES=1,2,3 '
    elif exc_gpu == 2:
        extra = 'CUDA_VISIBLE_DEVICES=4,5,6,7 '
    elif exc_gpu == 3:
        extra = 'CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 '
    else:
        extra = ''
    return extra + 'nohup python train_classifier.py ' + ' '.join(args) + ' > py%d.out 2> py%d.err < /dev/null &' % (ix,ix)

def get_ordered_experiments():
    arg_list = []
    
    arg_list.append(['--N_train', '1000', '--noise_p', '0.3'])

    #number of training examples
    for n in range(7):
        for arch in ['film', 'cat', 'gan']:
            arg_list.append(['--N_train', str(500*2**n), '--u_arch', arch])
    
    #noise
    for n in np.linspace(.1,.5,5):
        for arch in ['film', 'cat', 'gan']:
            arg_list.append(['--noise_p', str(n), '--u_arch', arch])
        
    #arg_list.append(['--Y_fn', '%d+%dd%d' % (1, 0, 1), '--nU', '256', '--noise', '2'])
    #arg_list.append(['--Y_fn', '%d+%dd%d' % (0, 1, 16), '--nU', '1024', '--noise', '2'])
        
    #optional
    for u in range(2, 11, 2):
        nU = round(1.5**u)
        for mult in range(3, 12, 4):
            arg_list.append(['--Y_fn', '%d+%dd%d' % (1, mult, 1), '--nU', str(nU)])
            arg_list.append(['--Y_fn', '%d+%dd%d' % (mult, 1, 1), '--nU', str(nU)])
        for div in range(3, int(nU**.5), 4):
            arg_list.append(['--Y_fn', '%d+%dd%d' % (1, 1, div), '--nU', str(nU)])

    # visual complexity
    datasets = ['mnist', 'fmnist', 'cifar10', 'cifar100']#, 'svhn', 'cifar100']
    for ds in datasets:
        if 'cifar' in ds:
            arg_list.append(['--dataset', ds, '--img_only', '--arch', 'dense', '--bsz', '256'])
        else:
            arg_list.append(['--dataset', ds, '--img_only'])
        
    for ds in datasets:
        if 'cifar10' == ds:
            arg_list.append(['--dataset', ds, '--arch', 'dense', '--optim', 'nest'])
        elif 'cifar100' == ds:
            arg_list.append(['--dataset', ds, '--arch', 'dense', '--optim', 'nest', '--nU', '100', '--Y_fn', '%d+%dd%d' % (1, 1, 10)])
        else:
            arg_list.append(['--dataset', ds])
    
    # contextual complexity
    for u in range(2, 11, 2):
        nU = round(1.5**u)
        arg_list.append(['--Y_fn', '%d+%dd%d' % (1, 1, 1), '--nU', str(nU)])
    arg_list.append(['--Y_fn', '%d+%dd%d' % (1, 0, 1), '--nU', '256'])
    arg_list.append(['--Y_fn', '%d+%dd%d' % (0, 1, 16), '--nU', '1024'])
    

    
    ix = 0
    df = pd.read_csv(csv_path, index_col=0)
    while ix < len(arg_list):
        if main.get_args(arg_list[ix])['model_type'] in set(df['model_type']):
            arg_list.pop(ix);
        else:
            ix += 1
        

    return arg_list

def clean_df():
    df = pd.read_csv(csv_path, index_col=0)
    df = df[(df['acc'] != -1) & (df['epochs'] > 10)]
    
    for fn in os.listdir('results'):
        if fn[:fn.find('.')] not in df.index:
            os.remove('results/'+fn)
            
    for Fn in glob.glob('data/*/*.npy'):
        fn = basename(Fn)
        if fn[:fn.rfind('_')] not in set(df['model_type']):
            os.remove(Fn)
            
    df.to_csv(csv_path)

    
