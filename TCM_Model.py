import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import *
import numpy as np
import torch
import torch.nn.functional as F

from torch.nn.init import kaiming_normal_,constant_

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()

    def forward(self, x):
        return x.view(x.size(0),-1)

class En(nn.Module):

    def __init__(self,in_dim, out_dim, Drop = False, p = 0.5, Isbasic = False,ly = [500,500,1000]):
        super(En, self).__init__()
        e1 = ly[0]
        e2 = ly[1]
        e3 = ly[2]
        z = out_dim
        d1 = e3
        d2 = e2
        d3 = e1
        self.Drop = Drop
        self.p = p
        self.basic = Isbasic
        if not Isbasic:
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, e1),
                nn.BatchNorm1d(e1),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(e1, e2),
                nn.BatchNorm1d(e2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(e2, e3),
                nn.BatchNorm1d(e3),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(e3, z),
            )
            self.encoder.apply(init_weights)

        else:
            self.basic_cls = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, out_dim),
            )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, Drop = True):
        if self.basic:
            logit_z = self.basic_cls(x)
            z = self.softmax(logit_z)
            return z,logit_z
        else:
            if Drop:
                if self.Drop:
                    x = F.dropout(x,self.p)
            logit_z = self.encoder(x)
            z = self.softmax(logit_z)

            return z, logit_z

class De(nn.Module):

    def __init__(self,in_dim, out_dim,ly = [500,500,1000]):
        super(De, self).__init__()
        e1 = ly[0]
        e2 = ly[1]
        e3 = ly[2]
        z = out_dim
        d1 = e3
        d2 = e2
        d3 = e1

        self.decoder = nn.Sequential(
            nn.Linear(z, d1),
            nn.BatchNorm1d(d1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(d1, d2),
            nn.BatchNorm1d(d2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(d2, d3),
            nn.BatchNorm1d(d3),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(d3, in_dim),
        )
        self.decoder.apply(init_weights)

    def forward(self, x):
        decode = self.decoder(x)

        return decode

class CLS(nn.Module):
    def __init__(self, in_dim, out_dim, bottle_neck_dim = 500):
        super(CLS, self).__init__()
        if bottle_neck_dim:
            self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
            self.fc = nn.Linear(bottle_neck_dim, out_dim)
            self.main = nn.Sequential(
                self.bottleneck,
                nn.Sequential(
                    nn.BatchNorm1d(bottle_neck_dim),
                    nn.LeakyReLU(0.2, inplace = True),
                    self.fc
                ),
                nn.Softmax(dim = -1)
            )
        else:
            self.fc = nn.Linear(in_dim, out_dim)
            self.main = nn.Sequential(
                self.fc,
                nn.Softmax(dim = -1)
            )

    def forward(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out

if __name__ == '__main__':
    from torchsummary import summary
    model = En(2048,31).cuda()
    summary(model,(1,2048))