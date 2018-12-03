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
    if slurm:
        ix = 0
        while exists("/data/vision/polina/users/clintonw/code/vision_final/scripts/srun%d.out" % ix):
            ix += 1
        return 'nohup ./srunner.sh cls %d ' % n_gpus + ' '.join(args) + ' > srun%d.out 2> srun%d.err < /dev/null &' % (ix,ix)

    ix = 0
    while exists("/data/vision/polina/users/clintonw/code/vision_final/scripts/py%d.out" % ix):
        ix += 1
    if exc_gpu == 1:
        extra = 'CUDA_VISIBLE_DEVICES=1,2,3 '
    elif exc_gpu == 2:
        extra = 'CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 '
    else:
        extra = ''
    return extra + 'nohup python train_classifier.py ' + ' '.join(args) + ' > py%d.out 2> py%d.err < /dev/null &' % (ix,ix)

def get_ordered_experiments():
    arg_list = []

    for n in range(6):
        #arg_list.append(['--N_train', str(60000//2**n), '--mc_drop', '0.3'])
        arg_list.append(['--N_train', str(60000//2**n), '--mc_drop', '0.3', '--arch', 'dense'])
    for n in np.round(np.linspace(0,.5,6), 1):
        #arg_list.append(['--noise_p', str(n), '--mc_drop', '0.3'])
        arg_list.append(['--noise_p', str(n), '--mc_drop', '0.3', '--arch', 'dense'])
    #, '--u_arch', 'film', '--arch', 'dense'
    #number of training examples
    #for n in range(6):
    #    arg_list.append(['--N_train', str(60000//2**n), '--arch', 'all-conv', '--u_arch', 'cat'])

    #noise
    #for n in np.round(np.linspace(0,.5,6), 1):
    #    for arch in ['dense', 'all-conv']:
    #        arg_list.append(['--noise_p', str(n), '--u_arch', 'cat', '--arch', arch])

    #optional
    """for u in range(2, 11, 2):
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
    arg_list.append(['--Y_fn', '%d+%dd%d' % (0, 1, 16), '--nU', '1024'])"""

    
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

    
