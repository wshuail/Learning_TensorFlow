#! /usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
# x = tf.placeholder(tf.float32, (None))
y_ = tf.placeholder(tf.float32, [None, 10])
# y_ = tf.placeholder(tf.int32, (None))

with tf.variable_scope('net'):
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    logits = tf.matmul(x, W) + b
    y = tf.nn.softmax(logits)

with tf.variable_scope('loss'):
    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_)
with tf.variable_scope('train'):
    train_op = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

with tf.variable_scope('eval'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_op, feed_dict={x: batch_xs, y_: batch_ys})
    if i % 100 == 0:
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

