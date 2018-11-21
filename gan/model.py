import logging
import matplotlib.pyplot as plt
import numpy as np
import numbers
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import types

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

class GANModel(nn.Module):

    BATCH_SIZE = 64
    SHUFFLE = True
    GPU_AVAILABLE = torch.cuda.is_available()
    G_LEARNING_RATE = 0.0002
    D_LEARNING_RATE = 0.0002
    G_BETA1 = 0.5
    D_BETA1 = 0.5
    Z_DIM = 100
    NUM_EPOCH=20
    CHECKPOINT_INTERVAL=1

    def __init__(self, G, D, 
                model_path,
                logger, # TODO: Make this optional by creating an empty base logging class
                z_dim=Z_DIM,
                num_epoch=NUM_EPOCH,
                checkpoint_interval=CHECKPOINT_INTERVAL,
                dataset=None,
                batch_size=BATCH_SIZE,
                shuffle=SHUFFLE,
                g_lr=G_LEARNING_RATE,
                d_lr=D_LEARNING_RATE,
                g_beta1=G_BETA1,
                d_beta1=D_BETA1,
                use_cuda=GPU_AVAILABLE):
        super(GANModel, self).__init__()
        self._G = G
        self._D = D
        self._z_dim = z_dim
        self._model_path = model_path
        self._logger = logger
        self._checkpoint_interval = checkpoint_interval
        
        self._num_epoch = num_epoch
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle

        self._g_lr = g_lr
        self._d_lr = d_lr
        self._g_beta1 = g_beta1
        self._d_beta1 = d_beta1
        
        self._criterion = nn.BCELoss()
        
        self._use_cuda = use_cuda
        self._init_data_loader()
        self._init_optimizers()

        if self._use_cuda:
            self._G = self._G.cuda()
            self._D = self._D.cuda()

    @property
    def z_dim(self):
        return self._z_dim

    @z_dim.setter
    def z_dim(self, value):
        if not value.is_integer() or value <= 0:
            raise ValueError('z_dim must be a positive integer')
        self._z_dim = int(value)

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        if not isinstance(value, Dataset):
            raise ValueError('The dataset should of type torch.utils.data.Dataset')
        self._dataset = value
        self._init_data_loader()
        logging.info('Reinitialized the data loader with the updated batch size property')

    @property
    def model_path(self):
        return self._model_path

    @model_path.setter
    def model_path(self, value):
        if not os.path.exists(value):
            os.makedirs(value)
            logging.info('Created Directory: %s' % value)
        self._model_path = value
    
    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, value):
        path = os.path.join(self._logger, value.id)
        if not os.path.exists(path):
            os.makedirs(path)
            logging.info('Created Directory: %s' % path)
        self._logger = value

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if not value.is_integer() or value <= 0:
            raise ValueError('batch_size must be a positive integer')
        self._batch_size = int(value)
        self._init_data_loader()
        logging.info('Reinitialized the data loader with the updated batch size property')

    @property
    def shuffle(self):
        return self._shuffle
    
    @shuffle.setter
    def shuffle(self, value):
        if type(value) != types.BooleanType:
            raise ValueError('shuffle must be a boolean')
        self._shuffle = value
        self._init_data_loader()
        logging.info('Reinitialized the data loader with the updated shuffle property')

    @property
    def g_lr(self):
        return self._g_lr

    @g_lr.setter
    def g_lr(self, value):
        if not isinstance(value, numbers.Number) or value <= 0:
            raise ValueError('g_lr must be a positive number')
        self._g_lr = value
        self._init_optimizers(g=True, d=False)

    @property
    def d_lr(self):
        return self._d_lr

    @d_lr.setter
    def d_lr(self, value):
        if not isinstance(value, numbers.Number) or value <= 0:
            raise ValueError('d_lr must be a positive number')
        self._d_lr = value
        self._init_optimizers(g=False, d=True)

    @property
    def g_beta1(self):
        return self._g_lr

    @g_beta1.setter
    def g_beta1(self, value):
        if not isinstance(value, numbers.Number) or value <= 0 or value > 1:
            raise ValueError('g_beta1 must be a number between 0 and 1')
        self._g_beta1 = value
        self._init_optimizers(g=True, d=False)

    @property
    def d_beta1(self):
        return self._g_lr

    @d_beta1.setter
    def d_beta1(self, value):
        if not isinstance(value, numbers.Number) or value <= 0 or value > 1:
            raise ValueError('d_beta1 must be a number between 0 and 1')
        self._d_beta1 = value
        self._init_optimizers(g=False, d=True)



    def _init_optimizers(self, g=True, d=True):
        # TODO: support any optimizer
        if g:
            self._d_optimizer = optim.Adam(self._D.parameters(), lr=self._d_lr, betas=(self._d_beta1, 0.999))
        if d:
            self._g_optimizer = optim.Adam(self._G.parameters(), lr=self._g_lr, betas=(self._g_beta1, 0.999))

    def _init_data_loader(self):
        self._data_loader = DataLoader(dataset=self._dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            drop_last=True)

    def _step(self, x):
        if self._use_cuda:
            x = x.cuda()
        
        # 1. Train D on real+fake
        self._D.zero_grad()

        # 1A: Train D on real
        d_real_data = Variable(x)
        d_real_decision = self._D(d_real_data)
        d_real_labels = torch.ones(self._batch_size) # ones = true
        if self._use_cuda:
            d_real_labels = d_real_labels.cuda()
        d_real_error = self._criterion(d_real_decision, Variable(d_real_labels))  
        d_real_error.backward() # compute/store gradients, but don't change params

        #  1B: Train D on fake
        d_z = torch.randn(self._batch_size, self._z_dim)
        if self._use_cuda:
            d_z = d_z.cuda()
        d_gen_input = Variable(d_z)
        d_fake_data = self._G(d_gen_input).detach()  # detach to avoid training G on these labels
        d_fake_decision = self._D(d_fake_data)
        d_fake_labels = torch.zeros(self._batch_size) # zeros = fake
        if self._use_cuda:
            d_fake_labels = d_fake_labels.cuda()
        d_fake_error = self._criterion(d_fake_decision, Variable(d_fake_labels)) 
        d_fake_error.backward()
        self._d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()

        # 2. Train G on D's response (but DO NOT train D on these labels)
        self._G.zero_grad()

        g_z = torch.randn(self._batch_size, self._z_dim)
        if self._use_cuda:
            g_z = g_z.cuda()
        gen_input = Variable(g_z)
        g_fake_data = self._G(gen_input)
        dg_fake_decision = self._D(g_fake_data)
        dg_fake_labels = torch.ones(self._batch_size)
        if self._use_cuda:
            dg_fake_labels = dg_fake_labels.cuda()
        g_error = self._criterion(dg_fake_decision, Variable(dg_fake_labels))  # we want to fool, so pretend it's all genuine

        g_error.backward()
        self._g_optimizer.step()  # Only optimizes G's parameters
        
        return (d_real_error, d_fake_error, g_error)

    def _extract(self, v):
        return v.data.storage().tolist()

    def sample(self, n):
        z = torch.randn(n, self._z_dim)
        if self._use_cuda:
            z = z.cuda()
        x = self._G(z).detach()
        if self._use_cuda:
            x = x.cpu()
        return x

    def train(self):
        # TODO: Logging Abstraction. Add a layer of indirection for logging.
        #       Currently using comet.ml experiment directly
        with self._logger.train():
            for epoch in range(self._num_epoch):
                d_real_error, d_fake_error, g_error = None, None, None
                for i, (x, _) in enumerate(self._data_loader):
                    d_real_error, d_fake_error, g_error = self._step(x)
                    self._logger.log_metric('d_real_error', self._extract(d_real_error)[0], step=i)
                    self._logger.log_metric('d_fake_error', self._extract(d_fake_error)[0], step=i)
                    self._logger.log_metric('g_error', self._extract(g_error)[0], step=i)
                
                if epoch % self._checkpoint_interval == 0:
                    torch.save({
                        'epoch': epoch,
                        'g_model': self._G.state_dict(),
                        'd_model': self._D.state_dict(),
                        'g_optim': self._g_optimizer.state_dict(),
                        'd_optim': self._d_optimizer.state_dict(),
                        'd_real_error': self._extract(d_real_error)[0],
                        'd_fake_error': self._extract(d_fake_error)[0],
                        'g_error': self._extract(g_error)[0]
                    }, '%s.pt' % os.path.join(self._model_path, self._logger.id, epoch))
                    
                    samples = self.sample(self._batch_size)
                    w = h = int(self._batch_size ** 0.5)
                    fig = plt.figure(figsize=(w, h))
                    plt.axis('off')
                    for i in range(1, w * h + 1):
                        fig.add_subplot(w,h,i)
                        img = samples[i - 1].numpy()
                        img = (img + 1) / 2 * 255
                        img = np.moveaxis(img, 0, 2)
                        if img.shape[2] == 1:
                            img = img.reshape(img.shape[0], img.shape[1])
                        plt.imshow(img)
                    self._logger.log_figure(figure_name='epoch-%s' % epoch)
                
                self._logger.log_epoch_end(epoch)