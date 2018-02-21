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
        # self.scale = nn.Parameter(torch.FloatTensor([255]))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.mul(self.sigmoid(x), 255.0).view(-1, 1, 28, 28)

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
        return self.sigmoid(x)

generator = Generator()
discriminator = Discriminator()

generator_optim = optim.SGD(generator.parameters(), lr = 0.01, momentum=0.9)
discriminator_optim = optim.SGD(discriminator.parameters(), lr = 0.01, momentum=0.9)

def generator_loss(z):
    x_ = generator(z)
    d_ = discriminator(x_)
    loss = torch.add(torch.neg(d_), 1)
    log_loss = torch.log(loss)
    return torch.div(torch.sum(log_loss), z.size(0))

def discriminator_loss(x, z):
    d = discriminator(x)
    loss_d = torch.div(torch.sum(torch.log(d)), x.size(0))
    loss_g = generator_loss(z)
    return torch.add(loss_d, loss_g)

def train_discriminator(x, z):
    discriminator.train()
    discriminator_optim.zero_grad()
    loss = discriminator_loss(x, z)
    loss.backward()
    discriminator_optim.step()

def train_generator(z):
    generator.train()
    generator_optim.zero_grad()
    loss = generator_loss(x, z)
    loss.backward()
    generator_optim.step()

def train_epoch():
    for batch_idx, (x, _) in enumerate(train_loader):
        x = Variable(x)
        train_discriminator(x, Variable(torch.randn(x.size(0), 784)))
        if (batch_idx + 1) % k == 0:
            train_generator(Variable(torch.randn(x.size(0), 784)))

def train():
    for e in range(EPOCH):
        train_epoch()

train()
img = generator(Variable(torch.randn(1, 784)))
ans = (img.data.numpy().reshape(28, 28))
plt.imshow(ans, cmap='gray')
plt.show()