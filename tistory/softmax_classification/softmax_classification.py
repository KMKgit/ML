import tensorflow as tf
import numpy as np

xy = np.loadtxt('./data/softmax_train.txt', unpack=True, dtype='float32')
x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])

X = tf.placeholder("float", [None, 3]) # [1, x1, x2] 1 is bias
Y = tf.placeholder("float", [None, 3]) # [A,  B,  C] 3 classes

W = tf.Variable(tf.zeros([3, 3]))

hypothesis = tf.nn.softmax(tf.matmul(X, W)) 
# why not (W, X)?
# if X is 2D tensor with multiple inputs,
# 
# (X, W) is trick

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), reduction_indices=1))

a = tf.Variable(0.001)

optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:   #sess = tf.Session()
    sess.run(init)

    for step in xrange(2001):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step % 200 == 0:
            print (step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))
    
    #arg_max is one-hot encoding
    a = sess.run(hypothesis, feed_dict={X:[[1, 11, 7]]})
    print a, sess.run(tf.arg_max(a, 1))
    
    b = sess.run(hypothesis, feed_dict={X:[[1, 3, 4]]})
    print b, sess.run(tf.arg_max(b, 1))
    
    c = sess.run(hypothesis, feed_dict={X:[[1, 1, 0]]})
    print c, sess.run(tf.arg_max(c, 1))
    