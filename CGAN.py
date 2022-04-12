import random
import torch
from torch import nn, optim
import numpy as np
import os
import struct
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torchvision import datasets, transforms

import torch.nn.functional as F

batchsz = 128


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
        x = x.view(batchsz, 128, 3, 3)
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
        x = self.fc1(label).view(batchsz, 1, 28, 28)
        x = torch.cat([image, x], dim=1)
        x = self.net(x)
        return x


def weights_init(m):
    """
    Initialise weights of the model.
    """
    if type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def main():
    torch.manual_seed(23)
    np.random.seed(23)

    D = Discriminator().cuda()
    G = Generator().cuda()
    D.apply(weights_init)
    G.apply(weights_init)

    BCE_loss = nn.BCELoss()

    optim_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optim_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform),
        batch_size=128, shuffle=True)

    for epoch in range(102):
        # learning rate decay
        if (epoch + 1) == 11:
            optim_G.param_groups[0]['lr'] /= 10
            optim_D.param_groups[0]['lr'] /= 10
            print("learning rate change!")

        if (epoch + 1) == 16:
            optim_G.param_groups[0]['lr'] /= 10
            optim_D.param_groups[0]['lr'] /= 10
            print("learning rate change!")

        for image, label in train_loader:
            if batchsz != int(image.size()[0]):
                continue
            # Train Discriminator
            image = torch.FloatTensor(image).cuda().view(batchsz, 1, 28, 28)
            label = label.type(torch.LongTensor).cuda()

            y_real_ = torch.ones(batchsz).cuda()
            y_fake_ = torch.zeros(batchsz).cuda()
            y_label_ = torch.zeros(batchsz, 10).cuda()
            y_label_.scatter_(1, label.view(batchsz, 1), 1)

            # train discriminator D
            optim_D.zero_grad()
            D_result = D(image, y_label_).squeeze()
            D_real_loss = BCE_loss(D_result, y_real_)       # real loss

            z_ = torch.randn((batchsz, 100)).cuda()
            y_ = (torch.rand(batchsz, 1) * 10).type(torch.LongTensor).squeeze().cuda()
            y_label_ = torch.zeros(batchsz, 10).cuda()
            y_label_.scatter_(1, y_.view(batchsz, 1), 1)

            G_result = G(z_, y_label_)
            D_result = D(G_result, y_label_).squeeze()
            D_fake_loss = BCE_loss(D_result, y_fake_)       # fake loss

            D_train_loss = D_real_loss + D_fake_loss
            D_train_loss.backward()
            optim_D.step()

            # train generator G
            optim_G.zero_grad()

            z_ = torch.randn((batchsz, 100)).cuda()
            y_ = (torch.rand(batchsz, 1) * 10).type(torch.LongTensor).squeeze().cuda()
            y_label_ = torch.zeros(batchsz, 10).cuda()
            y_label_.scatter_(1, y_.view(batchsz, 1), 1)

            G_result = G(z_, y_label_)
            D_result = D(G_result, y_label_).squeeze()
            G_train_loss = BCE_loss(D_result, y_real_)

            G_train_loss.backward()
            optim_G.step()

        if epoch % 10 == 0:
            print("Epoch:%d, LossD:%.4f, LossG:%.4f." % (epoch+1, D_train_loss, G_train_loss))
            with torch.no_grad():
                z_ = torch.randn((batchsz, 100)).cuda()
                y_ = (torch.rand(batchsz, 1) * 10).type(torch.LongTensor).squeeze().cuda()
                y_label_ = torch.zeros(batchsz, 10).cuda()
                y_label_.scatter_(1, y_.view(batchsz, 1), 1)
                gen_data = G(z_, y_label_).detach().cpu()
            plt.figure(figsize=(10, 10))
            plt.axis("off")
            plt.imshow(np.transpose(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True), (1, 2, 0)))
            plt.pause(0.5)
            plt.savefig("epoch%d.jpg" % epoch)
            plt.close('all')
            torch.save(G.state_dict(), os.path.join(os.getcwd(), 'gen%d.pth' % epoch))
            torch.save(D.state_dict(), os.path.join(os.getcwd(), 'dis%d.pth' % epoch))

    torch.save(G.state_dict(), os.path.join(os.getcwd(), 'gen.pth'))
    torch.save(D.state_dict(), os.path.join(os.getcwd(), 'dis.pth'))


if __name__ == "__main__":
    main()
    plt.show()

