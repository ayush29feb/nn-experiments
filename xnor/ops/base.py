import numpy as np

from abc import ABCMeta, abstractmethod

class BaseOp:
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward(self, x, params=None):
        pass
    
    @abstractmethod
    def backward(self, df):
        pass

    @abstractmethod
    def get_initial_params(self, input_shape):
        pass
    
    @abstractmethod
    def get_output_shape(slef, input_shape):
        pass