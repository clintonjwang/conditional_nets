import numpy as np
import pandas as pd
import scipy.stats as stats
import os, glob
from os.path import *
import scipy.ndimage.filters as filters

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Ellipse
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib as mpl
import seaborn as sns

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count() - 2


"""def heatmap(data, sigma=15, cmap='hot'):
    data = filters.gaussian_filter(data, sigma=sigma)
    #colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0.75), (0, 1, 0), (0.75, 1, 0),
    #          (1, 1, 0), (1, 0.8, 0), (1, 0.7, 0), (1, 0, 0)]
    #cm = LinearSegmentedColormap.from_list('sample', colors)

    plt.imshow(data, cmap=cmap) #cm
    plt.colorbar()"""

def heatmap(data, sigma=15, cmap='hot'):
    plt.imshow(ents, cmap='hot', interpolation='nearest')

    
"""
=========
POSTERIORS
=========
"""

def get_accs(train_zuy, val_zuy, args):
    emp_post = get_emp_post(train_zuy, nY=args['nY'])
    true_post = get_true_post(args)
    emp_estimator = np.argmax(emp_post, axis=0)
    true_estimator = np.argmax(true_post, axis=0)
    emp_acc = get_post_acc(emp_estimator, val_zuy)
    true_acc = get_post_acc(true_estimator, val_zuy)

    return emp_acc, true_acc

def get_post_acc(estimator, zuy):
    return (estimator[zuy[:,0], zuy[:,1]] == zuy[:,2]).sum().float().item()/len(zuy)

def get_stats(zuy, args, preds):
    emp_post = get_emp_post(zuy, nY=args['nY'])
    true_post = get_true_post(args)
    pred_post, pZ, pU = get_pred_post(preds, zuy[:,:2])
    true_KL = kl_div(pred_post, true_post, pZ, pU)
    emp_KL = kl_div(pred_post, emp_post, pZ, pU)
    true_JS = js_div(pred_post, true_post, pZ, pU)
    emp_JS = js_div(pred_post, emp_post, pZ, pU)
    
    return true_KL, emp_KL, true_JS, emp_JS
    
def get_emp_post(zuy, nY):
    """True image labels, context and outcome (N*3)
    Returns empirical posterior (nZ*nU*n_cls)"""
    nZ = zuy[:,0].max()+1
    nU = zuy[:,1].max()+1
    pZ = np.zeros(nZ, dtype=float)
    pU = np.zeros(nU, dtype=float)
    pY_ZU = np.zeros((nY,nZ,nU), dtype=float)
    
    for i in range(len(zuy)):
        pZ[zuy[i,0]] += 1
        pU[zuy[i,1]] += 1
        
    pZ /= pZ.sum()
    pU /= pU.sum()
    
    for i in range(len(zuy)):
        pY_ZU[zuy[i,-1],zuy[i,0],zuy[i,1]] += 1/(pZ[zuy[i,0]]*pU[zuy[i,1]])
    
    pY_ZU /= pY_ZU.sum(0, keepdims=True)
        
    return pY_ZU
    
def get_true_post(args):
    """Returns true posterior (nZ*nU*n_cls)"""
    f = args['f']
    nZ = args['nZ']
    nU = args['nU']
    pY_ZU = np.zeros((args['nY'],nZ,nU))
    for x in range(nZ):
        for u in range(nU):
            pY_ZU[f(x,u),x,u] = 1-args['noise_p']
            for n in range(1, args['noise_lim']+1):
                pY_ZU[f(x,u)+n,x,u] = args['noise_p']/args['noise_lim']
    return pY_ZU
    
def get_pred_post(pred, xu):
    """Predicted outcomes (N*n_cls)
    True image labels and context (N*2)
    Returns predicted posterior (n_cls*nZ*nU), as well as pZ and pU"""
    nZ = xu[:,0].max()+1
    nU = xu[:,1].max()+1
    nY = pred.shape[1]
    pZ = np.zeros(nZ, dtype=float)
    pU = np.zeros(nU, dtype=float)
    pY_ZU = np.zeros((nY,nZ,nU), dtype=float)
    
    for i in range(len(xu)):
        pZ[xu[i,0]] += 1
        pU[xu[i,1]] += 1
        
    pZ /= pZ.sum()
    pU /= pU.sum()
    
    for i in range(len(xu)):
        for y in range(nY):
            pY_ZU[y,xu[i,0],xu[i,1]] += pred[i,y]/(pZ[xu[i,0]]*pU[xu[i,1]])
    
    pY_ZU /= pY_ZU.sum(0, keepdims=True)
        
    return pY_ZU, pZ, pU
    
def kl_div(post1, post2, pZ, pU):
    """Posteriors of size (n_cls*nZ*nU)"""
    return np.nanmax([0,(post1 * pZ.reshape(1,-1,1) * pU.reshape(1,1,-1) * np.log(post1/(post2+1e-6))).sum()])
    
def js_div(post1, post2, pZ, pU):
    """Posteriors of size (n_cls*nZ*nU)"""
    M = (post1+post2)/2
    pZU = pZ.reshape(1,-1,1) * pU.reshape(1,1,-1)
    return np.nanmax([0,(post1 * pZU * np.log(post1/(M+1e-6))).sum()]) + np.nanmax(
                        [0,(post2 * pZU * np.log(post2/(M+1e-6))).sum()]) 
    
"""def kl_div(pred_post, true_post, emp_post, pZ, pU):
    #Predicted posterior (n_cls*nZ*nU)
    #True posterior (n_cls*nZ*nU)
    #Empirical posterior (n_cls*nZ*nU)
    true_KL = (pred_post * pZ.reshape(1,-1,1) * pU.reshape(1,1,-1) * np.log(pred_post/(true_post+1e-6))).sum()
    emp_KL = (pred_post * pZ.reshape(1,-1,1) * pU.reshape(1,1,-1) * np.log(pred_post/(emp_post+1e-6))).sum()
    return true_KL, emp_KL"""
    
    
def get_entropy(args):
    """f is function of 2 vars, e.g. f = lambda x,u: (x+u)//3
    noise must be uniform and discrete over its range
    """
    nZ = args['nZ']
    nU = args['nU']
    f = args['f']

    nY = f(nZ-1,nU-1)+1+args['noise_lim']
    pZ = np.ones(nZ)/nZ

    if args['context_dist'] == 'uniform':
        pU_X = np.ones((nU,nZ))/nU

    elif args['context_dist'] == 'binomial':
        pU_X = np.zeros((nU,nZ))
        for x in range(nZ):
            pU_X[:,x] = stats.binom.pmf(range(nU), n=nU, p=(x+1)/(nZ+1) )
            
        pU_X /= pU_X.sum(0)

    pZU = (pZ * pU_X).transpose()
    pU = pZU.sum(0)

    pY = np.zeros(nY)
    pY_ZU = np.zeros((nY,nZ,nU))
    pYXU = np.zeros((nY,nZ,nU))
    for x in range(nZ):
        for u in range(nU):
            pY_ZU[f(x,u),x,u] = 1-args['noise_p']
            for n in range(1, args['noise_lim']+1):
                pY_ZU[f(x,u)+n,x,u] = args['noise_p']/args['noise_lim']

            for n in range(args['noise_lim']+1):
                pY[f(x,u)+n] += pZU[x,u]
                pYXU[f(x,u)+n,x,u] = np.expand_dims(pZU[x,u], 0) * pY_ZU[f(x,u)+n,x,u]

    pY_X = (pY_ZU*pU.reshape((1,1,-1))).sum(2)
    pY_U = (pY_ZU*pZ.reshape((1,-1,1))).sum(1)

    pYX = pY_X * pZ.reshape((1,-1))
    pYU = pY_U * pU.reshape((1,-1))

    HX = np.nansum(-pZ*np.log(pZ))
    HU = np.nansum(-pU*np.log(pU))
    HY = np.nansum(-pY*np.log(pY))
    HY_X = np.nansum(-pYX*np.log(pY_X))
    HY_U = np.nansum(-pYU*np.log(pY_U))
    HY_XU = np.nansum(-pYXU*np.log(pY_ZU))
    IXU = np.nansum(pZU*(np.log(pZU) - np.log(pZ).reshape((-1,1)) - np.log(pU).reshape((1,-1)) ))

    entropies = {'HZ': HX, 'HU': HU, 'HY': HY, 'HY_Z': HY_X, 'HY_U': HY_U, 'HY_ZU': HY_XU, 'IZU': IXU}

    return entropies
