import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

# constants
NHIDDEN = 24
STDEV = 0.5
KMIX = 24
NOUT = 3 * KMIX
NSAMPLE = 1000
NEPOCH = 10000

# initialize data placeholders
x = tf.placeholder(dtype=tf.float32, shape=(None, 1))
y = tf.placeholder(dtype=tf.float32, shape=(None, 1))

# hidden layer
Wh = tf.Variable(tf.random_normal((1, NHIDDEN), stddev=STDEV, dtype=tf.float32))
bh = tf.Variable(tf.random_normal((1, 1), stddev=STDEV, dtype=tf.float32))
h = tf.nn.tanh(tf.matmul(x, Wh) + bh)

# out layer
Wo = tf.Variable(tf.random_normal((NHIDDEN, NOUT), stddev=STDEV, dtype=tf.float32))
bo = tf.Variable(tf.random_normal((1, NOUT), stddev=STDEV, dtype=tf.float32))
o = tf.matmul(h, Wo) + bo

def get_mixture_coeff(o):
    """
    Returns the 
    """
    o_pi = tf.placeholder(dtype=tf.float32, shape=(None, KMIX), name='mixparam')
    o_sigma = tf.placeholder(dtype=tf.float32, shape=(None, KMIX), name='mixparam')
    o_mu = tf.placeholder(dtype=tf.float32, shape=(None, KMIX), name='mixparam')

    o_pi, o_sigma, o_mu = tf.split(o, [KMIX, KMIX, KMIX], 1)

    max_pi = tf.reduce_max(o_pi, 1, keep_dims=True)
    o_pi = tf.subtract(o_pi, max_pi)
    o_pi = tf.exp(o_pi)
    o_pi = tf.multiply(o_pi, tf.reciprocal(tf.reduce_sum(o_pi, 1, keep_dims=True)))

    o_sigma = tf.exp(o_sigma)
    
    return o_pi, o_sigma, o_mu

# initialize data
y_data = np.float32(np.random.uniform(-10.5, 10.5, (1, NSAMPLE))).T
r_data = np.float32(np.random.normal(size=(NSAMPLE, 1)))
x_data = np.float32(np.sin(0.75 * y_data) * 7.0 + y_data * 0.5 + r_data * 1.0) # x = 7sin(3y/4) + y/2 + r

plt.figure(figsize=(8, 8))
plt.plot(x_data, y_data, 'ro', alpha=0.2)
plt.show()

# cost function
def tf_normal(y, sigma, mu):
    """
    return e ** (-0.5 * ((y - mu) / sigma) ** 2) / 2 * pi * sigma
    """
    result = tf.subtract(y, mu)
    result = tf.multiply(result, tf.reciprocal(sigma))
    result = -tf.square(result) / 2
    result = tf.multiply(tf.exp(result), tf.reciprocal(sigma)) * (1 / math.sqrt(2 * math.pi))
    return result

def lossfunc(y, pi, sigma, mu):
    """
    loss = -log(sum_k (pi_k * normal(y, mu, sigma)))
    """
    result = tf_normal(y, sigma, mu)
    result = tf.multiply(result, pi)
    result = tf.reduce_sum(result, 1, keep_dims=True)
    result = -tf.log(result)
    result = tf.reduce_mean(result)
    return result

# loss and train op
o_pi, o_sigma, o_mu = get_mixture_coeff(o)
loss = lossfunc(y, o_pi, o_sigma, o_mu)
train_op = tf.train.AdamOptimizer().minimize(loss)

# create tf session
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# train model
losses = np.zeros(NEPOCH)
for i in range(NEPOCH):
    sess.run(train_op, feed_dict={ x: x_data, y: y_data })
    losses[i] = sess.run(loss, feed_dict={x: x_data, y: y_data})

plt.figure(figsize=(8, 8))
plt.plot(np.arange(100, NEPOCH, 1), losses[100:], 'r-')
plt.show()

# generate samples from the learned parameters
x_test = np.float32(np.arange(-15,15,0.1))
NTEST = x_test.size
x_test = x_test.reshape(NTEST,1) # needs to be a matrix, not a vector

def get_pi_idx(x, pdf):
  N = pdf.size
  accumulate = 0
  for i in range(0, N):
    accumulate += pdf[i]
    if (accumulate >= x):
      return i
  print 'error with sampling ensemble'
  return -1

def generate_ensemble(out_pi, out_mu, out_sigma, M = 10):
  NTEST = x_test.size
  result = np.random.rand(NTEST, M) # initially random [0, 1]
  rn = np.random.randn(NTEST, M) # normal random matrix (0.0, 1.0)
  mu = 0
  std = 0
  idx = 0

  # transforms result into random ensembles
  for j in range(0, M):
    for i in range(0, NTEST):
      idx = get_pi_idx(result[i, j], out_pi[i])
      mu = out_mu[i, idx]
      std = out_sigma[i, idx]
      result[i, j] = mu + rn[i, j]*std
  return result

o_pi_test, o_sigma_test, o_mu_test = sess.run(get_mixture_coeff(o), feed_dict={x: x_test})

y_test = generate_ensemble(o_pi_test, o_mu_test, o_sigma_test)

plt.figure(figsize=(8, 8))
plt.plot(x_data,y_data,'ro', x_test,y_test,'bo',alpha=0.3)
plt.show()