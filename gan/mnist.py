import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.autograd import Variable

BATCH_SIZE = 32
DATA_DIR = 'data'
EPOCH = 10
k = 2

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(DATA_DIR, 
            train=True, 
            download=True, 
            transform=transforms.Compose([
                transforms.ToTensor()
            ])),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(DATA_DIR, 
            train=False, 
            download=True, 
            transform=transforms.Compose([
                transforms.ToTensor()
            ])),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(784, 784)
        self.fc2 = nn.Linear(784, 784)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x.view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=3, padding=1)
        self.fc = nn.Linear(5 * 28 * 28, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.fc(x.view(x.size(0), -1))
        x = self.sigmoid(x)
        return x

generator = Generator()
discriminator = Discriminator()

generator_optim = optim.SGD(generator.parameters(), lr = 0.0001, momentum=0.9)
discriminator_optim = optim.SGD(discriminator.parameters(), lr = 0.0001, momentum=0.9)

BCE_loss = nn.BCELoss()

generator_losses = []
discriminator_losses = []

def train_discriminator(x, z):
    discriminator_optim.zero_grad()
    d, d_ = discriminator(x), discriminator(generator(z))
    y, y_ = Variable(torch.ones(z.size(0))), Variable(torch.zeros(z.size(0)))
    loss = BCE_loss(d, y) + BCE_loss(d_, y_)
    discriminator_losses.append(loss.data[0])
    loss.backward()
    discriminator_optim.step()

def train_generator(z):
    generator_optim.zero_grad()
    d_ = discriminator(generator(z))
    y_ = Variable(torch.zeros(z.size(0)))
    loss = BCE_loss(d_, y_)
    generator_losses.append(loss.data[0])
    loss.backward()
    generator_optim.step()

def train_epoch():
    for batch_idx, (x, _) in enumerate(train_loader):
        train_discriminator(Variable(x), Variable(torch.randn(x.size(0), 1, 28, 28)))
        if (batch_idx + 1) % k == 0:
            train_generator(Variable(torch.randn(x.size(0), 1, 28, 28)))

def train():
    generator.train()
    discriminator.train()
    for e in range(EPOCH):
        train_epoch()
        plt.plot(generator_losses, 'g')
        plt.plot(discriminator_losses[::k], 'b')
        plt.savefig('results/loss_%d.png' % e)
        plt.close()
        img = generator(Variable(torch.randn(1, 1, 28, 28)))
        plt.imshow(img.data.numpy().reshape(28, 28) * 255, cmap='gray')
        plt.savefig('results/test_%d.png' % e)
        plt.close()

train()