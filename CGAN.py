import random
import torch
from torch import nn, optim
import numpy as np
import os
import struct
import matplotlib.pyplot as plt
import torchvision.utils as vutils

import torch.nn.functional as F

batchsz = 128


def load_mnist(path, kind='train'):
    """Load MNIST data from `path` https://www.cnblogs.com/xianhan/p/9145966.html"""
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)     # .
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def data_generator(imgs, labs, kind='train'):
    """get a batch of data"""
    while True:
        img = []
        lab = []
        for _ in range(batchsz):
            if kind == 'train':
                i = random.randint(0, 60000 - 1)
            else:
                i = random.randint(0, 10000 - 1)
            img.append(imgs[i])
            lab.append(labs[i])
        img = np.array(img, dtype='uint8')
        lab = np.array(lab, dtype='uint8')
        yield img, lab


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


# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()   # floor( (size-kernel+2padding)/stride ) + 1
#         self.net = nn.Sequential(
#             # in [1, 28, 28]
#             nn.Conv2d(1, 64, 4, 2, 1),  # [64, 14, 14]
#             nn.LeakyReLU(0.1, True),  # overwrite the data
#             nn.Conv2d(64, 128, 4, 2, 1),  # [128, 7, 7]
#             nn.LeakyReLU(0.1, True),
#             nn.Conv2d(128, 512, 7),  # [512, 1, 1]
#             nn.LeakyReLU(0.1, True),  # linear layer
#         )
#         self.fc1 = nn.Sequential(
#             nn.Linear(10, 512),
#             nn.LeakyReLU(0.1, True)
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear(1024, 128),
#             nn.LeakyReLU(0.1, True),
#             nn.Linear(128, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, image, label):
#         x = self.net(image).view(batchsz, -1)
#         y = self.fc1(label)
#         x = torch.cat([x, y], 1)
#         x = self.fc2(x)
#         return x


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

    img, lab = load_mnist(os.path.join(os.getcwd(), 'mnist'))
    data_itr = data_generator(img, lab)
    for epoch in range(8002):
        # Train Discriminator
        image, label = next(data_itr)
        image = torch.FloatTensor(image).cuda().view(batchsz, 1, 28, 28)
        image = (image - 127.5)/127.5
        label = torch.from_numpy(label).type(torch.LongTensor).cuda()

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

        if epoch % 100 == 0:
            print("Epoch:%d, LossD:%.4f, LossG:%.4f." % (epoch+1, D_train_loss, G_train_loss))
            with torch.no_grad():
                gen_data = G(z_, y_label_).detach().cpu()
            plt.figure(figsize=(10, 10))
            plt.axis("off")
            plt.imshow(np.transpose(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True), (1, 2, 0)))
            plt.pause(0.5)
            plt.savefig("epoch%d.jpg" % epoch)
            plt.close('all')

    torch.save(G.state_dict(), os.path.join(os.getcwd(), 'gen.pth'))
    torch.save(D.state_dict(), os.path.join(os.getcwd(), 'dis.pth'))


if __name__ == "__main__":
    main()
    plt.show()

