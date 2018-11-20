import matplotlib.pyplot as plt
import numpy as np

def imgrid(x):
    """
    A simple method to visualize a batch of images.

    x: (N, C, W, H)
    """
    N = x.shape[0]
    w = h = int(N ** 0.5)
    fig = plt.figure(figsize = (w, h))
    plt.axis('off')

    for i in range(1, w * h + 1):
        fig.add_subplot(w, h, i)
        img = x[i - 1].numpy()
        img = np.moveaxis(img, 0, 2)
        plt.imshow(img)