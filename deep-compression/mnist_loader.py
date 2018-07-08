import cPickle
import gzip
import logging
import numpy as np
import os
import wget

DATA_URL = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
FILENAME = 'mnist.pkl.gz'

def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]

class DataLoader:

    def __init__(self, dataset='training', batch_size=32, dirpath='data/'):
        self.dataset = dataset
        self.batch_size = batch_size
        self.cursor = 0
        
        if not os.path.exists(os.path.join(dirpath, FILENAME)):
            os.makedirs(dirpath)
            wget.download(DATA_URL, out=os.path.join(dirpath, FILENAME))

        data_file = gzip.open(os.path.join(os.curdir, 'data', 'mnist.pkl.gz'), 'rb')
        self.training_data, self.validation_data, self.test_data = cPickle.load(data_file)
        data_file.close()

    def next_batch(self):
        if self.dataset is 'training':
            img = [np.reshape(x, (28, 28)) for x in self.training_data[0]]
            lbl = [y for y in self.training_data[1]]
        elif self.dataset is 'testing':
            img = [np.reshape(x, (28, 28)) for x in self.test_data[0]]
            lbl = [y for y in self.test_data[1]]
        elif self.dataset is 'validation':
            img = [np.reshape(x, (28, 28)) for x in self.validation_data[0]]
            lbl = [y for y in self.validation_data[1]]
        else:
            raise ValueError, 'Invalid Dataset Name'

        img, lbl = np.array(img), np.array(lbl)

        start = self.cursor
        self.cursor += self.batch_size

        if self.cursor < len(lbl):
            return (
                img[start:start+self.batch_size], 
                lbl[start:start+self.batch_size]
            )
        else:
            self.cursor = self.cursor % len(lbl)
            return (
                np.append(img[start:], img[:self.cursor]),
                np.append(lbl[start:], lbl[:self.cursor]),
            )

    def reset_dataset(self, dataset='train'):
        self.cursor = 0
        self.dataset = dataset
