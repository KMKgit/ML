import tensorflow as tf
import numpy as np
import random
#import imp
#load = imp.load_source('load', './load.py')
from tensorflow.examples.tutorials.mnist import input_data

X = tf.placeholder("float", [None, 784]) # [1, x1, x2] 1 is bias
Y = tf.placeholder("float", [None, 10]) # [A,  B,  C] 3 classes

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), reduction_indices=1))

a = tf.Variable(0.001)

optimizer = tf.train.GradientDescentOptimizer(a).minimize(cost)
init = tf.initialize_all_variables()

n_epoch = 25
n_step = 1
batch_size = 100
mnist = input_data.read_data_sets("MNIST_data", one_hot = True)


with tf.Session() as sess:   #sess = tf.Session()
    sess.run(init)
    
    for epoch in range(n_epoch):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={X: batch_xs, Y:batch_ys})
        
        avg_cost += sess.run(cost, feed_dict={X: batch_xs, Y:batch_ys}) / total_batch
        if epoch % n_step == 0:
            print ("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
            print (sess.run(b))

    print ("Learning Finished!")
    
    # Get one & predict now
    r = random.randint(0, mnist.test.num_examples - 1)
    print ("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print ("Prediction:", sess.run(tf.argmax(hypothesis,1), {X: mnist.test.images[r:r+1]}))
    
    # Test model
    pred = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    
    # Calculate accuracy
    acc = tf.reduce_mean(tf.cast(pred, "float"))
    print ("Acc:", acc.eval({X: mnist.test.images, Y: mnist.test.labels}))
