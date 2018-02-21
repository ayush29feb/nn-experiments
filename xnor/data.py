import numpy as np
import math

def fake_data(N):
    x_train, y_train = np.empty((0, 2)), []

    # Class 0
    theta = np.random.rand(N) * math.pi * 2
    r = np.random.rand(N) * 50
    x = np.array([r * np.cos(theta), r * np.sin(theta)]).T
    x_train = np.vstack((x_train, x))
    y_train.extend([0] * N)

    # Class 1-4
    for i in range(1, 5):
        theta = np.random.rand(N) * math.pi / 2 + (i * math.pi / 2)
        r = np.random.rand(N) * 50 + 50
        x = np.array([r * np.cos(theta), r * np.sin(theta)]).T
        x_train = np.vstack((x_train, x))
        y_train.extend([i] * N)

    y_train = np.array(y_train)
    return x_train, y_train