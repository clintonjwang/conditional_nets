import torch
import torch.nn.functional as F
import niftiutils.nn.submodules as subm
import math
nn = torch.nn

class BaseCNN(nn.Module):
    def __init__(self, n_h, dims=(1,28,28), n_cls=10):
        super(BaseCNN, self).__init__()
        self.dims = dims
        self.n_h = n_h
        self.conv = nn.Sequential(
            nn.Conv2d(dims[0], 64, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            #nn.Conv2d(64, 64, kernel_size=3, padding=1),
            #nn.Dropout2d(.2),
            #nn.LeakyReLU(.1,True),
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
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)


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
        super(FilmCNN, self).__init__(dims=dims, n_h=args['h_dim'], n_cls=nZ)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.u_type = args['u_arch']
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
            nn.Linear(128, nY)
        )
        self.film = nn.Sequential(
            nn.Linear(self.n_h, 128),
            nn.Dropout(.2),
            nn.ReLU(True),
            nn.Linear(128, nY)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x, u):
        h = self.conv(x)
        h = self.global_pool(h).view(h.size(0), -1)
        
        u = self.pre_film(u)
        if self.u_type == 'cat':
            uh = torch.cat([u, h], 1)
            y = self.cat(uh)
        else:
            uh = torch.mul(u[:,:self.n_h], h) + u[:,self.n_h:]
            y = self.film(uh)
        return y


class FilmAE(FilmCNN):
    def __init__(self, args, **kwargs):
        super(FilmAE, self).__init__(args, **kwargs)
        self.recon_fc = nn.Sequential(
            nn.Linear(args['h_dim'], 1024),
            nn.Dropout(.2, True),
            nn.ReLU(),

            nn.Linear(1024, 128*7*7),
            nn.Dropout(.2, True),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            # [-1, 128, 7, 7] -> [-1, 64, 14, 14]
            nn.ConvTranspose2d(128,64,4,2,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # -> [-1, 1, 28, 28]
            nn.ConvTranspose2d(64,1,4,2,1),
            nn.Sigmoid()
        )
        
        for m in self.recon_fc:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, u):
        h = self.conv(x)
        h = self.global_pool(h).view(h.size(0), -1)

        u = self.pre_film(u)
        if self.u_type == 'cat':
            uh = torch.cat([u, h], 1)
            y = self.cat(uh)
        else:
            uh = torch.mul(u[:,:self.n_h], h) + u[:,self.n_h:]
            y = self.film(uh)

        z = self.recon_fc(h)
        z = z.view(-1, 128, 7, 7)
        recon = self.deconv(z)*1.1-.05

        return y, recon
