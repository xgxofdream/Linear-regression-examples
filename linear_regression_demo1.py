from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
rng = np.random

# Parameters.
learning_rate = 0.01
training_steps = 1000
display_step = 50

# Training Data.
X = np.array([1,2,3,4,5,6,7,8,9])
Y = np.array([1,2,3,4,5,6,7,8,9])
n_samples = X.shape[0]

# Weight and Bias, initialized randomly.
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Linear regression (Wx + b).
def linear_regression(x):
    return W * x + b

# Mean square error.
def mean_square(y_pred, y_true):
    return tf.reduce_sum(tf.pow(y_pred-y_true, 2)) / (2 * n_samples)

# Stochastic Gradient Descent Optimizer.
optimizer = tf.optimizers.SGD(learning_rate)

# Optimization process. 
def run_optimization():
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        pred = linear_regression(X)
        loss = mean_square(pred, Y)

# Compute gradients.
    gradients = g.gradient(loss, [W, b])
    
# Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, [W, b]))
    
# Run training for the given number of steps.
for step in range(1, training_steps + 1):
# Run the optimization to update W and b values.
    run_optimization()
    
    if step % display_step == 0:
        pred = linear_regression(X)
        loss = mean_square(pred, Y)
        print("step: %i, loss: %f, W: %f, b: %f" % (step, loss, W.numpy(), b.numpy()))
import matplotlib.pyplot as plt

# Graphic display
plt.plot(X, Y, 'ro', label='Original data')
plt.plot(X, np.array(W * X + b), label='Fitted line')
plt.legend()
plt.show()