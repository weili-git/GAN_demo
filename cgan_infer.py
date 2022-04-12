import random
import torch
from torch import nn, optim
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision.utils as vutils


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()   # n` = (n-1)s + k - 2p
        self.fc1 = nn.Linear(10, 50, bias=False)
        self.fc2 = nn.Sequential(
            nn.Linear(150, 3 * 3 * 128),
            nn.BatchNorm1d(3 * 3 * 128),
            nn.ReLU()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 0, bias=False),  # [64, 7, 7]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),  # [32, 14, 14]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),  # [1, 28, 28]
            nn.Tanh()
        )

    def forward(self, z, label):
        x = self.fc1(label)
        x = torch.cat([z, x], 1)
        x = self.fc2(x)
        x = x.view(128, 128, 3, 3)
        x = self.net(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()   # floor( (size-kernel+2padding)/stride ) + 1
        self.fc1 = nn.Linear(10, 28 * 28)
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.5),    # avoid over fitting
            nn.Conv2d(32, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(128*7*7, 1),
            nn.Sigmoid()
        )

    def forward(self, image, label):
        x = self.fc1(label).view(128, 1, 28, 28)
        x = torch.cat([image, x], dim=1)
        x = self.net(x)
        return x


G = Generator().cuda()
G.load_state_dict(torch.load(os.path.join(os.getcwd(), 'gen.pth')))

z_ = torch.randn((128, 100)).cuda()
y_ = (torch.rand(128, 1) * 10).type(torch.LongTensor).squeeze().cuda()

i = 0
for c in '31415926':
    y_[i] = int(c)
    i = i + 10

y_label_ = torch.zeros(128, 10).cuda()
y_label_.scatter_(1, y_.view(128, 1), 1)

with torch.no_grad():
    gen_data = G(z_, y_label_).detach().cpu()

plt.figure(figsize=(10, 10))
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True), (1, 2, 0)))
plt.show()