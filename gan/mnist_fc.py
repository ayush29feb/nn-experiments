import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.autograd import Variable

from gan import GAN
from zoo import *

DATA_DIR='data'
BATCH_SIZE=128
Z_DIM=50
IMG_DIM = 28 * 28
NUM_EPOCH=10

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

gan = GAN(Generator(), Discriminator(), train_loader, num_epoch=NUM_EPOCH)
gan.train()
torch.save(gan, 'gan-10.pt')
generated_sampels = gan.sample(np.random.randn(10, 50))
for i in range(generated_sampels.shape[0]):
    plt.imshow(generated_sampels[i, 0], cmap='gray')
    plt.savefig('sample_%d.png' % i)
