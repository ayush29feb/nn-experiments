import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.autograd import Variable

BATCH_SIZE = 50
DATA_DIR = 'data'

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(DATA, 
            train=True, 
            download=True, 
            transform=transforms.Compose([
                transforms.ToTensor()
            ])),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(DATA, 
            train=False, 
            download=True, 
            transform=transforms.Compose([
                transforms.ToTensor()
            ])),
    batch_size=BATCH_SIZE,
    shuffle=True,
)