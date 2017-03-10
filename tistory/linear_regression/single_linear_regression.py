import tensorflow as tf

x_data = [0, 1, 2, 1000]
y_data = [0, 0, 1, 1]

# tf.random_uniform([1], -1.0, 1.0) 
# -> create [1] value between -1.0 to 1.0 based random
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# H(x) = Wx + b
hypothesis = W * x_data + b

# reduce_mean() -> mean
# cost is operation
# not execute calculator in here(line 17)
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# Minimize
a = tf.Variable(0.1) # Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(2001):
    sess.run(train)
    if step % 20 == 0:
        print (step, sess.run(cost), sess.run(W), sess.run(b))
