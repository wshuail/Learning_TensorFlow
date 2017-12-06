#! /usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

n_steps = 28
n_inputs = 28
n_neurons = 150
n_outputs = 10

learning_rate = 0.001

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
# y = tf.placeholder(tf.int32, [None])
y = tf.placeholder(tf.int32, [None, n_outputs])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, x, dtype=tf.float32)

# logits = tf.contrib.layers.fully_connected(states, n_outputs, activation_fn=None)

logits = tf.layers.dense(inputs=states, units=n_outputs,
                         activation=None,
                         kernel_initializer=tf.random_uniform_initializer(0., 0.1),
                         bias_initializer=tf.constant_initializer(0.1),
                         name='logits')

xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)

loss = tf.reduce_mean(xentropy)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)

# correct = tf.nn.in_top_k(logits, y, 1)
correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)
x_test = mnist.test.images.reshape(-1, n_steps, n_inputs)
y_test = mnist.test.labels

n_epochs = 100
batch_size = 150

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x_batch, y_batch = None, None
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            x_batch = x_batch.reshape(-1, n_steps, n_inputs)
            sess.run(train_op, feed_dict={x: x_batch, y: y_batch})

        if epoch % 10 == 0:
            acc_train = sess.run(accuracy, feed_dict={x: x_batch, y: y_batch})
            acc_test = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
            print (epoch, ' Training accuracy: ', acc_train, ' Test accuracy: ', acc_test)




