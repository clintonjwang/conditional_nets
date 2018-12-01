#https://github.com/bamos/densenet.pytorch
import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import sys
import math

class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    drop = nn.Dropout2d(.5)
    
    def __init__(self, dims=(1,28,28), growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=10):
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(dims[0], nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)
        self.nChannels = nChannels

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.drop(out)
        out = self.trans2(self.dense2(out))
        out = self.drop(out)
        out = self.dense3(out)
        out = self.drop(out)
        out = torch.squeeze(F.adaptive_avg_pool2d(F.relu(self.bn1(out)), 1))
        out = self.fc(out)
        return out
    
    

class FilmDenseNet(DenseNet):
    def __init__(self, args, **kwargs):
        super(FilmDenseNet, self).__init__(nClasses=args['nZ'], **kwargs)
        self.u_type = args['u_arch']
        self.fc = nn.Linear(self.nChannels, args['h_dim'])
        self.n_h = args['h_dim']
        # Linear layer
        self.pre_film = nn.Sequential(
            nn.Linear(args['nU'], 128),
            nn.Dropout(.2),
            nn.ReLU(True),
            nn.Linear(128, self.n_h*2)
        )
        self.cat = nn.Sequential(
            nn.Linear(self.n_h*3, 128),
            nn.Dropout(.2),
            nn.ReLU(True),
            nn.Linear(128, args['nY'])
        )
        self.film = nn.Sequential(
            nn.Linear(self.n_h, 128),
            nn.Dropout(.2),
            nn.ReLU(True),
            nn.Linear(128, args['nY'])
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, u):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        h = torch.squeeze(F.adaptive_avg_pool2d(F.relu(self.bn1(out)), 1))
        h = self.fc(h)

        u = self.pre_film(u)
        if self.u_type == 'cat':
            uh = torch.cat([u, h], 1)
            y = self.cat(uh)
        else:
            uh = torch.mul(u[:,:self.n_h], h) + u[:,self.n_h:]
            y = self.film(uh)
        return y


class FilmDenseAE(FilmDenseNet):
    def __init__(self, args, **kwargs):
        super(FilmDenseNet, self).__init__(args, **kwargs)
        self.recon_fc = nn.Sequential(
            nn.Linear(z_dim + cc_dim + dc_dim, 1024),
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            nn.Linear(1024, 128*7*7),
            nn.BatchNorm2d(128*7*7),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            # [-1, 128, 7, 7] -> [-1, 64, 14, 14]
            nn.ConvTranspose2d(128,64,4,2,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # -> [-1, 1, 28, 28]
            nn.ConvTranspose2d(64,1,4,2,1),
            nn.Tanh()
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, u):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        h = torch.squeeze(F.adaptive_avg_pool2d(F.relu(self.bn1(out)), 1))
        h = self.fc(h)

        u = self.pre_film(u)
        if self.u_type == 'cat':
            uh = torch.cat([u, h], 1)
            y = self.cat(uh)
        else:
            uh = torch.mul(u[:,:self.n_h], h) + u[:,self.n_h:]
            y = self.film(uh)

        z = self.recon_fc(h)
        z = z.view(-1, 128, 7, 7)
        recon = self.deconv(z)

        return y, recon
