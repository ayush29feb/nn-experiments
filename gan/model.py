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
    LOGGING_STEP_SIZE=10

    def __init__(self, G, D, 
                model_path,
                logger, # TODO: Make this optional by creating an empty base logging class
                z_dim=Z_DIM,
                num_epoch=NUM_EPOCH,
                checkpoint_interval=CHECKPOINT_INTERVAL,
                logging_step_size=LOGGING_STEP_SIZE,
                dataset=None,
                batch_size=BATCH_SIZE,
                shuffle=SHUFFLE,
                g_lr=G_LEARNING_RATE,
                d_lr=D_LEARNING_RATE,
                g_beta1=G_BETA1,
                d_beta1=D_BETA1,
                sample_size=64,
                use_cuda=GPU_AVAILABLE):
        super(GANModel, self).__init__()
        self._G = G
        self._D = D
        self._z_dim = z_dim
        self.model_path = model_path
        self.logger = logger
        self._checkpoint_interval = checkpoint_interval
        self._logging_step_size = logging_step_size
        self._sample_size = sample_size
        
        self._num_epoch = num_epoch
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle

        self._g_lr = g_lr
        self._d_lr = d_lr
        self._g_beta1 = g_beta1
        self._d_beta1 = d_beta1
        
        self._criterion = nn.BCELoss()
        
        self._device = torch.device("cuda:0" if use_cuda else "cpu")
        self._G.to(self._device)
        self._D.to(self._device)

        self._init_data_loader()
        self._init_optimizers()

        self._fixed_z = torch.randn(self._sample_size, self._z_dim, device=self._device)

        self._logger.log_multiple_params(locals())

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
        path = os.path.join(self._model_path, value.id)
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
            drop_last=False)

    def _step(self, x):
        
        # 1. Train D on real+fake
        self._D.zero_grad()
        real_data = x.to(self._device)
        batch_size = real_data.size(0)
        real_labels = torch.ones(batch_size, device=self._device) # ones = true
        fake_labels = torch.zeros(batch_size, device=self._device) # zeros = fake

        # 1A: Train D on real
        d_real_decision = self._D(real_data)
        d_real_error = self._criterion(d_real_decision, real_labels)
        d_real_error.backward() # compute/store gradients, but don't change params

        #  1B: Train D on fake
        d_z = torch.randn(batch_size, self._z_dim, deice=self._device)
        fake_data = self._G(d_z).detach()  # detach to avoid training G on these labels
        d_fake_decision = self._D(fake_data)
        d_fake_error = self._criterion(d_fake_decision, fake_labels)
        
        d_fake_error.backward()
        self._d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()

        # 2. Train G on D's response (but DO NOT train D on these labels)
        self._G.zero_grad()

        dg_fake_decision = self._D(fake_data)
        g_error = self._criterion(dg_fake_decision, fake_labels)  # we want to fool, so pretend it's all genuine
        g_error.backward()
        self._g_optimizer.step()  # Only optimizes G's parameters
        
        return (d_real_error.item(),
                d_fake_error.item(),
                g_error.item(),
                d_real_decision.mean().item(),
                d_fake_decision.mean().item(), 
                dg_fake_decision.mean().item())

    def sample(self, n):
        z = torch.randn(n, self._z_dim, device=self._device)
        x = self._G(z).detach()
        return x

    def train(self, start_epoch=0):
        # TODO: Logging Abstraction. Add a layer of indirection for logging.
        #       Currently using comet.ml experiment directly
        with self._logger.train():
            for epoch in range(start_epoch, self._num_epoch):
                d_real_error, d_fake_error, g_error = None, None, None
                for i, (x, _) in enumerate(self._data_loader):
                    step = len(self._data_loader) * epoch + i
                    d_real_error, d_fake_error, g_error, d_real_decision, d_fake_decision, dg_fake_decision = self._step(x)
                    if step % self._logging_step_size == 0:
                        self._logger.log_metric(d_real_error', d_real_error, step=step)
                        self._logger.log_metric('d_fake_error', d_fake_error, step=step)
                        self._logger.log_metric('g_error', g_error, step=step)
                        self._logger.log_metric('d_real_decision', d_real_decision, step=step)
                        self._logger.log_metric('d_fake_decision', d_fake_decision, step=step)
                        self._logger.log_metric('dg_fake_decision', dg_fake_decision, step=step)
                        self._logger.log_multiple_metrics(self._G.stats(), step=step)
                        self._logger.log_multiple_metrics(self._D.stats(), step=step)
                
                if epoch % self._checkpoint_interval == 0:
                    torch.save({
                        'epoch': epoch,
                        'g_model': self._G.state_dict(),
                        'd_model': self._D.state_dict(),
                        'g_optim': self._g_optimizer.state_dict(),
                        'd_optim': self._d_optimizer.state_dict(),
                        'd_real_error': d_real_error,
                        'd_fake_error': d_fake_error,
                        'g_error': g_error
                    }, os.path.join(self._model_path, self._logger.id, '%s.pt' % epoch))
                    
                    samples = self._G(self._fixed_z).detach().to('cpu')
                    w = h = int(self._sample_size ** 0.5)
                    fig = plt.figure(figsize=(w, h))
                    for i in range(1, w * h + 1):
                        fig.add_subplot(w,h,i)
                        img = samples[i - 1].numpy()
                        img = (img + 1) / 2 * 255
                        img = np.moveaxis(img, 0, 2)
                        if img.shape[2] == 1:
                            img = img.reshape(img.shape[0], img.shape[1])
                        plt.axis('off')
                        plt.imshow(img)
                    self._logger.log_figure(figure_name='epoch-%s' % epoch)
                
                self._logger.log_epoch_end(epoch)