import numpy as np
import scipy.stats as stats

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


def emp_post(xuy):
    """True image labels, context and outcome (N*3)
    Returns empirical posterior (nX*nU*n_cls)"""
    nX = xuy[:,0].max()+1
    nU = xuy[:,1].max()+1
    nY = xuy[:,-1].max()+1
    pX = np.zeros(nX, dtype=float)
    pU = np.zeros(nU, dtype=float)
    pY_XU = np.zeros((nY,nX,nU), dtype=float)
    
    for i in range(len(xuy)):
        pX[xuy[i,0]] += 1
        pU[xuy[i,1]] += 1
        
    pX /= pX.sum()
    pU /= pU.sum()
    
    for i in range(len(xuy)):
        pY_XU[xuy[i,-1],xuy[i,0],xuy[i,1]] += 1/(pX[xuy[i,0]]*pU[xuy[i,1]])
    
    pY_XU /= pY_XU.sum(0, keepdims=True)
        
    return pY_XU
    
def true_post(xu, f, noise=0):
    """True image labels and context (N*2)
    Function f(x,u)
    Noise n
    
    Returns true posterior (nX*nU*n_cls)"""
    nX = xu[:,0].max()+1
    nU = xu[:,1].max()+1
    nY = f(nU-1,nX-1)+1+noise
    pY_XU = np.zeros((nY,nX,nU))
    for x in range(nX):
        for u in range(nU):
            for n in range(noise+1):
                pY_XU[f(x,u)+n,x,u] = 1/(noise+1)
    return pY_XU
    
def pred_post(pred, xu):
    """Predicted outcomes (N*n_cls)
    True image labels and context (N*2)
    Returns predicted posterior (n_cls*nX*nU), as well as pX and pU"""
    nX = xu[:,0].max()+1
    nU = xu[:,1].max()+1
    nY = pred.shape[1]
    pX = np.zeros(nX, dtype=float)
    pU = np.zeros(nU, dtype=float)
    pY_XU = np.zeros((nY,nX,nU), dtype=float)
    
    for i in range(len(xu)):
        pX[xu[i,0]] += 1
        pU[xu[i,1]] += 1
        
    pX /= pX.sum()
    pU /= pU.sum()
    
    for i in range(len(xu)):
        for y in range(nY):
            pY_XU[y,xu[i,0],xu[i,1]] += pred[i,y]/(pX[xu[i,0]]*pU[xu[i,1]])
    
    pY_XU /= pY_XU.sum(0, keepdims=True)
        
    return pY_XU, pX, pU
    
def kl_div(post1, post2, pX, pU):
    """Posteriors of size (n_cls*nX*nU)"""
    return (post1 * pX.reshape(1,-1,1) * pU.reshape(1,1,-1) * np.log(post1/(post2+1e-6))).sum()
    
"""def kl_div(pred_post, true_post, emp_post, pX, pU):
    #Predicted posterior (n_cls*nX*nU)
    #True posterior (n_cls*nX*nU)
    #Empirical posterior (n_cls*nX*nU)
    true_KL = (pred_post * pX.reshape(1,-1,1) * pU.reshape(1,1,-1) * np.log(pred_post/(true_post+1e-6))).sum()
    emp_KL = (pred_post * pX.reshape(1,-1,1) * pU.reshape(1,1,-1) * np.log(pred_post/(emp_post+1e-6))).sum()
    return true_KL, emp_KL"""
    
    
def get_entropy(f, context_type='uniform', nU=20, nX=10, noise=0):
    """f is function of 2 vars, e.g. f = lambda x,u: (x+u)//3
    noise must be uniform and discrete over its range
    """
    
    nY = f(nU-1,nX-1)+1+noise
    pX = np.ones(nX)/nX

    if context_type == 'uniform':
        pU_X = np.ones((nU,nX))/nU

    elif context_type == 'binomial':
        pU_X = np.zeros((nU,nX))
        for x in range(nX):
            pU_X[:,x] = stats.binom.pmf(range(nU), n=nU, p=(x+1)/(nU-9) )

    pXU = (pX * pU_X).transpose()
    pU = pXU.sum(0)

    pY = np.zeros(nY)
    pY_XU = np.zeros((nY,nX,nU))
    for x in range(nX):
        for u in range(nU):
            for n in range(noise+1):
                pY[f(x,u)+n] += pXU[x,u]
                pY_XU[f(x,u)+n,x,u] = 1/(noise+1)
                pYXU[f(x,u)+n,x,u] = pX[x] * pU[u] * pY_XU[f(x,u)+n,x,u]

    pY_X = (pY_XU*pU.reshape((1,1,-1))).sum(2)
    pY_U = (pY_XU*pX.reshape((1,-1,1))).sum(1)

    pYX = pY_X * pX
    pYU = pY_U * pU

    HY = np.nansum(-pY*np.log(pY))
    HY_X = np.nansum(-pYX*np.log(pY_X))
    HY_U = np.nansum(-pYU*np.log(pY_U))
    HY_UX = 0

    IXU = np.nansum(pXU*(np.log(pXU) - np.log(pX).reshape((-1,1)) - np.log(pU).reshape((1,-1)) ))

    return HY_X, HY_U, HY, HY_UX, IXU