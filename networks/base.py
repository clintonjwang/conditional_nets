import torch
import torch.nn.functional as F
import niftiutils.nn.submodules as subm
import math
nn = torch.nn

class BaseCNN(nn.Module):
    def __init__(self, dims=(1,28,28), n_cls=10):
        super(BaseCNN, self).__init__()
        self.dims = dims
        self.n_h = 64
        self.conv = nn.Sequential(
            nn.Conv2d(dims[0], 64, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(64, self.n_h, kernel_size=5),
            nn.Dropout2d(.2),
            nn.MaxPool2d(2),
            nn.ReLU(True)
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )
        self.cls = nn.Linear(self.n_h, n_cls)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x).view(x.size(0), -1)
        z = self.cls(x)
        return z


class FilmCNN(BaseCNN):
    def __init__(self, args, dims=(1,28,28)):
        nZ=args['nZ']
        nU=args['nU']
        nY=args['nY']
        super(FilmCNN, self).__init__(dims=dims, n_cls=nZ)

        self.u_type = args['u_arch']
        self.pre_film = nn.Sequential(
            nn.Linear(nU, 64),
            nn.Dropout(.2),
            nn.ReLU(True)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.film = nn.Sequential(
            nn.Linear(self.n_h+64, 128),
            nn.Dropout(.2),
            nn.ReLU(True),
            nn.Linear(128, nY)
        )

    def forward(self, x, u):
        x = self.conv(x)
        x = self.global_pool(x).view(x.size(0), -1)
        u = self.pre_film(u)
        
        ux = torch.cat([u, x], 1)
        y = self.film(ux)
        return y
