import math
import numpy as np
import matplotlib.pyplot as plt

from data import fake_data
from model import Model
from ops import *

# Data
x_train, y_train = fake_data(1000)
x_test, y_test = fake_data(200)

# Network
model = Model((5000, 2))

model.add_layer(Dense2D(20))
model.add_layer(Relu())
model.add_layer(Dense2D(5))

model.initialize_params()

for i in range(1000):
    dx, loss = model.loss(x_train, y_train)
    print i, loss
    dparams = model.backward(dx)
    model.update_params(dparams)
