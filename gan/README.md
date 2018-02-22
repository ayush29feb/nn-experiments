# General Adversarial Networks

Implementation of a simple GAN trained on the classical MNIST handwritten digit dataset.

## Experiments
In order to gain a better understanding of how GANs behave with different G & D networks with various hyperparameters, I will perform a number of experiments. I will log the experiments results in this readme.

There are a number of hyper parameters and possible structures we can test
- batch size
- optim functions
-- adam
-- adagrad
-- sgd
- learning rate
- num epochs
- architecture
-- conv v/s fc
-- normalization
-- number of layers
-- leaky relu parameter
-- dropout parameter
- gan k value
- dimension of z