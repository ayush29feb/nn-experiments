import numpy as np
import matplotlib.pyplot as plt

class Model:

    def __init__(self, input_shape, layers=None):
        self.input_shape = input_shape
        self.layers = layers if layers is not None else []
        self.params = []

    def initialize_params(self):
        params = []
        input_shape = self.input_shape
        for op in self.layers:
            params.append(op.get_initial_params(input_shape))
            input_shape = op.get_output_shape(input_shape)
        
        self.params = params

    def add_layer(self, op):
        self.layers.append(op)
    
    def forward(self, x):
        x_ = x
        for op, params in zip(self.layers, self.params):
            x_ = op.forward(x_, params)
        return x_
    
    def backward(self, df):
        dparams = []
        df_ = df
        for op, params in reversed(zip(self.layers, self.params)):
            df_, dparams_ = op.backward(df_)
            dparams.append(dparams)
        
        return list(reversed(dparams))
    
    def loss(self, x, y):
        N = x.shape[0]

        scores = self.forward(x)
        scores = scores - np.max(scores, axis=1, keepdims=True)
        scores = np.exp(scores)
        probs = scores / np.sum(scores, axis=1, keepdims=True)

        loss = -np.sum(np.log(probs[:, y])) / N
        dx = probs.copy()
        dx[:, y] -= 1

        return dx, loss

    def update_params(self, dparams):
        for param, dparam in zip(self.params, dparams):
            if param is None or dparams is None:
                continue
            print type(param), type(dparam)
            for p, dp in zip(param, dparam):
                print type(p), type(dp)
                p -= dp * 0.01
            