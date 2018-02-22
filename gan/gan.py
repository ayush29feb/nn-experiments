import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.autograd import Variable

class GAN:

    def __init__(self, generator, discriminator, train_loader,
                generator_optim=None,
                discriminator_optim=None,
                dim_z=50,
                dim_out=784,
                k=1,
                batch_size=128,
                num_epoch=100,
                use_cuda=True):
        self.generator = generator
        self.discriminator = discriminator
        self.train_loader = train_loader
        self.dim_z = dim_z
        self.dim_out = dim_out
        self.k = k
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.use_cuda = use_cuda and torch.cuda.is_available()

        if self.use_cuda:
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()
        self.BCE_loss = nn.BCELoss()

        self.generator_optim = optim.Adam(self.generator.parameters(), lr = 0.000002) if generator_optim is None else generator_optim
        self.discriminator_optim = optim.Adam(self.discriminator.parameters(), lr = 0.0002) if discriminator_optim is None else discriminator_optim

        self.generator_losses = []
        self.discriminator_losses = []

    def train_discriminator(self, x, z):
        self.discriminator.train()
        self.generator.eval()
        self.discriminator_optim.zero_grad()
        d, d_ = self.discriminator(x), self.discriminator(self.generator(z))
        y, y_ = torch.ones(z.size(0), 1), torch.zeros(z.size(0), 1)
        if self.use_cuda:
            y, y_ = y.cuda(), y_.cuda()
        y, y_ = Variable(y), Variable(y_)
        loss = self.BCE_loss(d, y) + self.BCE_loss(d_, y_)
        loss.backward()
        self.discriminator_optim.step()
        return loss.data[0]
    
    def train_generator(self, z):
        self.discriminator.eval()
        self.generator.train()
        self.generator_optim.zero_grad()
        d_ = self.discriminator(self.generator(z))
        y_ = torch.ones(z.size(0), 1)
        if self.use_cuda:
            y_ = y_.cuda()
        y_ = Variable(y_)
        loss = self.BCE_loss(d_, y_)
        loss.backward()
        self.generator_optim.step()
        return loss.data[0]

    def train_epoch(self):
        d_loss, g_loss = [], []
        for batch_idx, (x, _) in enumerate(self.train_loader):
            z = torch.randn(x.size(0), self.dim_z)
            if self.use_cuda:
                x, z = x.cuda(), z.cuda()
            d_loss.append(self.train_discriminator(Variable(x), Variable(z)))
            if (batch_idx + 1) % self.k == 0:
                z = torch.randn(x.size(0), self.dim_z)
                if self.use_cuda:
                    z = z.cuda()
                g_loss.append(self.train_generator(Variable(z)))
        self.discriminator_losses.append(np.mean(np.array(d_loss)))
        self.generator_losses.append(np.mean(np.array(g_loss)))

    def train(self):
        for step in range(self.num_epoch):
            print 'Epoch %d' % step
            self.train_epoch()
    
    def get_losses(self):
        return (self.discriminator_losses, self.generator_losses)

    def sample(self, z):
        z = torch.FloatTensor(z)
        if self.use_cuda:
            z = z.cuda()
        z = Variable(z)
        return self.generator(z).cpu().data.numpy()
