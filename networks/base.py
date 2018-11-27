import torch
import torch.nn.functional as F
import niftiutils.nn.submodules as subm
nn = torch.nn

class BaseCNN(nn.Module):
    def __init__(self, dims=(1,28,28), n_cls=10):
        super(BaseCNN, self).__init__()
        self.dims = dims
        self.conv = nn.Sequential(
            nn.Dropout2d(.2),
            nn.Conv2d(dims[0], 96, kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(96, 96, kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(96, 96, kernel_size=3, stride=2),
            nn.Dropout2d(.5),
            nn.ReLU(True),
            nn.Conv2d(96, 192, kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(192, 192, kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(192, 192, kernel_size=3, stride=2),
            nn.Dropout2d(.5),
            nn.ReLU(True),
            nn.Conv2d(192, 192, kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(192, 192, kernel_size=1),
            nn.ReLU(True)
        )
        self.cls = nn.Sequential(
            nn.Conv2d(192, n_cls, kernel_size=1),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.cls(x).view(x.size(0), -1)
        return x


class FilmCNN(BaseCNN):
    def __init__(self, dims=(1,28,28), nZ=10, nU=10, nY=19):
        super(FilmCNN, self).__init__(dims=dims, n_cls=nZ)
        self.pre_film = nn.Sequential(
            nn.Linear(nU, 64),
            nn.Dropout(.2),
            nn.ReLU(True)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.film = nn.Sequential(
            nn.Linear(192+64, 128),
            nn.Dropout(.2),
            nn.ReLU(True),
            nn.Linear(128, nY)
        )

    def forward(self, x, u):
        x = self.conv(x)
        x = self.global_pool(x).view(x.size(0), -1)
        u = self.pre_film(u)
        
        ux = torch.cat([u, x], 1)
        ux = self.film(ux)
        return ux
