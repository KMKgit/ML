import tensorflow as tf
import numpy as np

xy = np.loadtxt('../data/linear_train.txt', delimiter=' ', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

print ('x', x_data)
print ('y', y_data)
#W = 1 by 3 array
W = tf.Variable(tf.random_uniform([1, len(x_data)], -5.0, 5.0))

# matrix multiplication (W, X) 
# tf.matmul(W, x_data) := tf.matmul(W, x_data) + b 
hypothesis = tf.matmul(W, x_data)

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(2001):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(cost), sess.run(W)
        
        