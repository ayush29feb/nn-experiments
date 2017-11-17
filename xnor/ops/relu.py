import numpy

from base_op import BaseOp

class Relu(BaseOp):

    def __init__(self, hidden_units):
        self.hidden_units = hidden_units
    
    def forward(self, x, params=None):
        self.x = x
        return x * (x > 0)
    
    def backward(self, dout):
        return dout * (self.x > 0)

    def get_initial_params(self, input_shape):
        return None