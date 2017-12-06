#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

x_data = np.linspace(1, 1000, 200).reshape([-1, 10])

x = tf.placeholder(tf.float32, shape=[None, 10], name='x')

k_init, b_init = tf.random_uniform_initializer(0., 0.2), tf.constant_initializer(0.2)
hidden_layer = tf.layers.dense(inputs=x,
                               units=64,
                               activation=tf.nn.relu,
                               kernel_initializer=k_init,
                               bias_initializer=b_init,
                               name='x')
softmax_output = tf.layers.dense(inputs=hidden_layer,
                                 units=2,
                                 activation=tf.nn.sigmoid,
                                 kernel_initializer=k_init,
                                 bias_initializer=b_init,
                                 name='softmax')

with tf.Session() as sess:
    for _ in range(100):
        sess.run(tf.global_variables_initializer())
        result = sess.run(softmax_output, feed_dict={x: x_data})
        print ('result: ', result)

print(np.sum(result, axis=1))