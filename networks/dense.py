import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.densenet import DenseNet

class FilmDenseNet(DenseNet):
    def __init__(self, nZ=10, nU=10, nY=19, **kwargs):
        super(FilmDenseNet, self).__init__(num_classes=nZ, **kwargs)
        self.u_type = args['u_arch']
        # Linear layer
        self.pre_film = nn.Sequential(
            nn.Linear(nU, 128),
            nn.Dropout(.2),
            nn.ReLU(True),
            nn.Linear(128, self.num_features*2)
        )
        self.cat = nn.Sequential(
            nn.Linear(self.num_features*3, 128),
            nn.Dropout(.2),
            nn.ReLU(True),
            nn.Linear(128, nY)
        )
        self.film = nn.Sequential(
            nn.Linear(self.num_features, 128),
            nn.Dropout(.2),
            nn.ReLU(True),
            nn.Linear(128, nY)
        )
        
        
        """# Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)"""

    def forward(self, x, u):
        features = self.features(x)
        z = F.relu(features, inplace=True)
        z = F.adaptive_avg_pool2d(z, 1).view(z.size(0), -1)

        u = self.pre_film(u)
        if self.u_type == 'cat':
            uz = torch.cat([u, z], 1)
            y = self.cat(uz)
        else:
            uz = torch.mul(u[:,:self.num_features], z) + u[:,self.num_features:]
            y = self.film(uz)
        return y

def densenet_cifar(depth, k, **kwargs):
    N = (depth - 4) // 6
    model = DenseNet(growth_rate=k, block_config=[N, N, N], num_init_features=2*k, **kwargs)
    return model

def film_densenet(depth, k, **kwargs):
    N = (depth - 4) // 6
    model = FilmDenseNet(growth_rate=k, block_config=[N, N, N], num_init_features=2*k, **kwargs)
    return model
