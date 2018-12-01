#https://github.com/taeoh-kim/Pytorch_InfoGAN
import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, z_dim=128, cc_dim=1, dc_dim=10):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim + cc_dim + dc_dim, 1024),
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            nn.Linear(1024, 128*7*7),
            nn.BatchNorm2d(128*7*7),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            # [-1, 128, 7, 7] -> [-1, 64, 14, 14]
            nn.ConvTranspose2d(128,64,4,2,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # -> [-1, 1, 28, 28]
            nn.ConvTranspose2d(64,1,4,2,1),
            nn.Tanh()
        )

    def forward(self, z):
        # [-1, z]
        z = self.fc( z )

        # [-1, 128*7*7] -> [-1, 128, 7, 7]
        z = z.view(-1, 128, 7, 7)
        out = self.conv(z)

        return out


class Discriminator(nn.Module):
    def __init__(self, cc_dim = 1, dc_dim = 10):
        super(Discriminator, self).__init__()
        self.cc_dim = cc_dim
        self.dc_dim = dc_dim

        self.conv = nn.Sequential(
            # [-1, 1, 28, 28] -> [-1, 64, 14, 14]
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),

            # [-1, 128, 7, 7]
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(128*7*7, 128),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 1 + cc_dim + dc_dim)
        )

    def forward(self, x):
        # -> [-1, 128*7*7]
        tmp = self.conv(x).view(-1, 128*7*7)

        # -> [-1, 1 + cc_dim + dc_dim]
        out = self.fc(tmp)

        # Discrimination Output
        out[:, 0] = F.sigmoid(out[:, 0].clone())

        # Continuous Code Output = Value Itself
        # Discrete Code Output (Class -> Softmax)
        out[:, self.cc_dim + 1:self.cc_dim + 1 + self.dc_dim] = F.softmax(out[:, self.cc_dim + 1:self.cc_dim + 1 + self.dc_dim].clone())

        return out
