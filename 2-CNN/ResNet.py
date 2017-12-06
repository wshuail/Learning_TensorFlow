#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/5 9:55
# @Author  : Wang Shuailong

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np


def batch_normal_relu(inputs, is_training, momentum=0.997, epsilon=1e-5):
    inputs = tf.layers.batch_normalization(inputs, momentum=momentum, epsilon=epsilon,
                                           training=is_training)
    inputs = tf.nn.relu(inputs)
    return inputs


def residual_block(inputs, output_channels, is_training, same_shape=True):
    strides = 1
    shortcut = inputs
    inputs = batch_normal_relu(inputs, is_training)
    if not same_shape:
        strides = 2
        shortcut = tf.layers.conv2d(inputs=inputs, filters=output_channels, kernel_size=1, strides=strides)
    inputs = tf.layers.conv2d(inputs=inputs, filters=output_channels, kernel_size=3, strides=strides, padding='SAME')
    inputs = batch_normal_relu(inputs, is_training)
    inputs = tf.layers.conv2d(inputs=inputs, filters=output_channels, kernel_size=3, strides=1, padding='SAME')

    outputs = shortcut + inputs
    return outputs


def build_resnet(inputs, n_outputs, is_traning):
    with tf.variable_scope('init_conv'):
        net = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=7, activation=None)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
    with tf.variable_scope('max_pooling'):
        net = tf.layers.max_pooling2d(net, pool_size=3, strides=2, padding='SAME')

    with tf.variable_scope('block_1'):
        net = residual_block(inputs=net, output_channels=64, is_training=is_traning)
        net = residual_block(inputs=net, output_channels=64, is_training=is_traning)

    with tf.variable_scope('block_2'):
        net = residual_block(inputs=net, output_channels=128, is_training=is_traning,
                             same_shape=False)
        net = residual_block(inputs=net, output_channels=128, is_training=is_traning)

    with tf.variable_scope('output'):
        net = batch_normal_relu(net, is_training=is_traning)
        net = tf.layers.average_pooling2d(inputs=net, pool_size=2, strides=1)
        net_shape = net.get_shape().as_list()
        net = tf.reshape(net, shape=[-1, np.prod(net_shape[1:])])
        net = tf.layers.dense(net, n_outputs, activation=tf.nn.softmax)
    return net


x_data = np.random.rand(10, 28, 28, 3).astype('float32')
x = tf.placeholder(tf.float32, shape=[None, 28, 28, 3], name='x')

# net = residual_block(x, output_channels=16, same_shape=False)
net = build_resnet(inputs=x, n_outputs=10, is_traning=False)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y = sess.run(net, feed_dict={x: x_data})
    print(y.shape)