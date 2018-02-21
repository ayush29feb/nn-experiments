import numpy as np

from base import BaseOp

class Relu(BaseOp):
    
    def forward(self, x, params=None):
        self.x = x
        return x * (x > 0)
    
    def backward(self, df):
        return (df * (self.x > 0), None)

    def get_initial_params(self, input_shape):
        return None

    def get_output_shape(self, input_shape):
        return input_shape