import numpy as np

from base import BaseOp

class Dense2D(BaseOp):

    def __init__(self, hidden_units):
        self.hidden_units = hidden_units
    
    def forward(self, x, params=None):
        assert params is not None

        w, b = params
        self.x = x
        self.w = w
        return np.dot(x, w) + b
    
    def backward(self, df):
        """
        Returns:
            dx: gradient for the input
            dparams: gradient for the respective params
        """
        D = self.hidden_units
        N = self.x.shape[0]
    
        dx = np.dot(df, self.w.T)
        dw = np.dot(self.x.T, df)
        db = np.dot(np.ones((D, N)), df)
        return (dx, (dw, db))

    def get_initial_params(self, input_shape):
        shape_w = (input_shape[1], self.hidden_units)
        shape_b = (self.hidden_units,)
        return (np.random.randn(*shape_w), np.random.randn(*shape_b))

    def get_output_shape(self, input_shape):
        return (input_shape[0], self.hidden_units)
