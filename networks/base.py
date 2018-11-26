import torch
import torch.nn.functional as F
import niftiutils.nn.submodules as subm
nn = torch.nn

class BaseCNN(nn.Module):
    def __init__(self, dims=(1,28,28), n_cls=10):
        super(BaseCNN, self).__init__()
        self.dims = dims
        self.z_dim = ((dims[1]//4) * (dims[2]//4) * 64)
        self.conv = nn.Sequential(
            nn.Conv2d(dims[0], 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(.2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(.2)
        )
        self.cls = nn.Sequential(
            nn.Linear(self.z_dim, 128),
            nn.Dropout(.2),
            nn.ReLU(True),
            nn.Linear(128, n_cls)
        )

    def forward(self, x):
        x = x.view(x.shape[0], *self.dims)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.cls(x)
        return x


class FilmCNN(BaseCNN):
    def __init__(self, n_cls=10, dims=(1,28,28), n_context=10, n_out=19):
        super(FilmCNN, self).__init__(dims=dims, n_cls=n_cls)
        self.pre_film = nn.Sequential(
            nn.Linear(n_context, 64),
            nn.Dropout(.2),
            nn.ReLU(True)
        )
        self.film = nn.Sequential(
            nn.Linear(self.z_dim+64, 128),
            nn.Dropout(.2),
            nn.ReLU(True),
            nn.Linear(128, n_out)
        )

    def forward(self, x, u):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        u = self.pre_film(u)
        
        ux = torch.cat([u, x], 1)
        ux = self.film(ux)
        return ux
