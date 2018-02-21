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
EPOCH = 1
k = 1

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

def generator_loss(z):
    x_ = generator(z)
    d_ = discriminator(x_)
    y = Variable(torch.zeros(z.size(0)))
    return BCE_loss(d_, y)

def discriminator_loss(x, z):
    d = discriminator(x)
    y = Variable(torch.ones(x.size(0)))
    loss_d = BCE_loss(d, y)
    loss_g = generator_loss(z)
    loss = torch.add(loss_d, loss_g)
    return loss

def train_discriminator(x, z):
    discriminator_optim.zero_grad()
    loss = discriminator_loss(x, z)
    discriminator_losses.append(loss.data[0])
    loss.backward()
    discriminator_optim.step()

def train_generator(z):
    generator_optim.zero_grad()
    loss = generator_loss(z)
    generator_losses.append(loss.data[0])
    loss.backward()
    generator_optim.step()

def train_epoch():
    for batch_idx, (x, _) in enumerate(train_loader):
        x = Variable(x)
        train_discriminator(x, Variable(torch.randn(x.size(0), 1, 28, 28)))
        if (batch_idx + 1) % k == 0:
            train_generator(Variable(torch.randn(x.size(0), 1, 28, 28)))
        
        if (batch_idx + 1) % 1000 == 0:
            img = generator(Variable(torch.randn(1, 1, 28, 28)))
            plt.imshow(img.data.numpy().reshape(28, 28) * 255, cmap='gray')
            plt.show()
        
        if (batch_idx + 1) % 100 == 0:
            plt.plot(generator_losses)
            plt.plot(discriminator_losses[::2])
            plt.show()

def train():
    generator.train()
    discriminator.train()
    for e in range(EPOCH):
        train_epoch()

train()