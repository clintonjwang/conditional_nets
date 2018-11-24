import gc
import sys
import glob, shutil, os
from os.path import *
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
torch.backends.cudnn.benchmark=True

sys.path.append('..')
import networks.base as nets
import niftiutils.nn.submodules as subm
import config

C = config.Config()

def get_accuracy(data_loader, par_model):
    corr, total = 0,1e-5
    for imgs, labels, fns in data_loader:
        shapes = [I.shape for I in imgs]
        max_size = np.max(shapes, 0)
        pads = [((max_size[1]-sh[1])//2, (max_size[1]-sh[1]+1)//2,
                 (max_size[0]-sh[0])//2, (max_size[0]-sh[0]+1)//2) for sh in shapes]
        pads = torch.tensor(pads, dtype=torch.long)
        imgs = torch.stack([F.pad(imgs[ix].permute((2,0,1)), pads[ix]) for ix in range(len(imgs))],0).float().cuda()
        labels = labels.cuda()

        pred = par_model(imgs)

        true_mod = labels[:,0]
        est = torch.max(pred, dim=1)[1]
        corr += (est == labels).sum()
        total += (true_mod >= 0).sum()

    return (corr.float()/total.float()).item()

def main(model_name):
    n_gpus = torch.cuda.device_count()
    batch_size = 128*n_gpus

    ds = dl.MNIST()
    train_loader, val_loader = ds.get_split_loaders(batch_size)

    N = len(ds)
    N_train = len(ds.train_indices)
    N_val = N - N_train

    model = nets.FilmCNN().cuda()
    par_model = torch.nn.DataParallel(model)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=-1).cuda()
    max_epochs = 200
    epoch = 1
    patience = 5
    loss_hist = [np.inf]*patience
    hist = {'loss': [], 'val-acc': []} #'train-mod': [], 'train-ax': [], 

    while epoch <= max_epochs:
        running_loss = 0.0
            
        print('Epoch: ' + str(epoch))
        model.train();
        par_model.train();

        for batch_num, (imgs, labels, fns) in enumerate(train_loader, 1):
            optimizer.zero_grad()

            shapes = [I.shape for I in imgs]
            max_size = np.max(shapes, 0)
            pads = [((max_size[1]-sh[1])//2, (max_size[1]-sh[1]+1)//2,
                     (max_size[0]-sh[0])//2, (max_size[0]-sh[0]+1)//2) for sh in shapes]
            pads = torch.tensor(pads, dtype=torch.long)
            imgs = torch.stack([F.pad(imgs[ix].permute((2,0,1)), pads[ix]) for ix in range(len(imgs))],0).float().cuda()
            labels = labels.cuda()

            pred = par_model(imgs)
            true_mod = labels[:,0]
            true_ax = labels[:,1]
            pred_mod = pred[:,:-3]
            pred_ax = pred[:,-3:]
            
            losses = []
            if (true_mod >= 0).sum().item() >= 0:
                losses.append(criterion(pred_mod, true_mod))
            elif (true_ax >= 0).sum().item() == 0:
                losses.append(criterion(pred_ax, true_ax) * np.log(C.n_mods)/np.log(3.))
            loss = sum(losses)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

        if running_loss < np.min(loss_hist):
            torch.save(model.state_dict(), "../history/%s.state" % (model_name))
            
        gc.collect()
        torch.cuda.empty_cache()
        model.eval();
        par_model.eval();

        print("Loss: %.2f" % (running_loss))
        
        acc = get_accuracy(val_loader, par_model)
        print("Validation accuracy: %.1f%%" % (100*acc))
        hist['loss'].append(running_loss)
        hist['val-acc'].append(acc)
            
        loss_hist.append(running_loss)
        if np.min(loss_hist[patience:]) >= loss_hist[-1-patience]:
            break
            
        gc.collect()
        epoch += 1

    with open("../history/%s.history" % model_name, 'wb') as f:
        pickle.dump(hist, f)
        

if __name__ == "__main__":
    model_name = sys.argv[1]
    
    main(model_name)