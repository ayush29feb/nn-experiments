import numpy

from base_op import BaseOp

class Dense2D(BaseOp):

    def __init__(self, hidden_units):
        self.hidden_units = hidden_units
    
    def forward(self, x, params=None):
        assert params is not None

        w, b = params
        self.x = x
        self.w = self.w
        return np.dot(x, w) + b
    
    def backward(self, dout):
        """
        Returns:
            dx: gradient for the input
            dparams: gradient for the respective params
        """
        D = self.b.shape.reshape(-1)[0]
        N = self.x.shape[0]
    
        dx = np.dot(dout, self.w.T)
        dw = np.dot(self.x.T, dout.T)
        db = np.dot(np.ones((D, N)), dout)
        return (dx, (dw, db))

    def get_initial_params(self, input_shape):
        shape_w = (input_shape[1], self.hidden_units)
        shape_b = (self.hidden_units)
        return (np.random.randn(shape_w), np.random.randn(shape_b))
