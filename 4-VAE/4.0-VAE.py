#! /usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../MNIST_data')

batch_size = 64

x = tf.placeholder(tf.float32, shape=[None, 28, 28], name='x')
y = tf.placeholder(tf.float32, shape=[None, 28, 28], name='y')
y_flat = tf.reshape(y, shape=[-1, 28*28])

keep_prob = tf.placeholder(tf.float32, shape=(), name='keep_prob')

dec_in_channels = 1
n_latent = 8

reshaped_dim = [-1, 7, 7, dec_in_channels]
inputs_decoder = 49*dec_in_channels//2

def leaky_relu(x, alpha=0.8):
    return tf.maximum(x, tf.multiply(x, alpha))

def encoder(x, keep_prob):
    activation = leaky_relu
    with tf.variable_scope('encoder', reuse=None):
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2,
                             padding='SAME', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2,
                             padding='SAME', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1,
                             padding='SAME', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)

        mn = tf.layers.dense(x, units=n_latent)
        sd = 0.5*tf.layers.dense(x, units=n_latent)
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent]))
        z = mn + tf.multiply(epsilon, tf.exp(sd))
        return z, mn, sd

def decoder(sampled_z, keep_prob):
    activation = leaky_relu
    with tf.variable_scope('decoder', reuse=None):
        x = tf.layers.dense(sampled_z, units=inputs_decoder, activation=activation)
        x = tf.layers.dense(x, units=inputs_decoder*2+1, activation=activation)
        x = tf.reshape(x, reshaped_dim)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2,
                                       padding='SAME', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1,
                                       padding='SAME', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1,
                                       padding='SAME', activation=tf.nn.relu)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, 28*28, activation=tf.nn.sigmoid)
        img = tf.reshape(x, shape=[-1, 28, 28])
        return img


sampled, mn, sd = encoder(x, keep_prob)
dec = decoder(sampled, keep_prob)

unshaped = tf.reshape(dec, [-1, 28*28])
img_loss = tf.reduce_sum(tf.squared_difference(unshaped, y_flat), 1)
latent_loss = -0.5*tf.reduce_sum(1.0 + 2.0*sd - tf.square(mn) - tf.exp(2.0*sd), 1)
loss = tf.reduce_mean(img_loss + latent_loss)
optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(30000):
    batch = [np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size=batch_size)[0]]
    sess.run(optimizer, feed_dict={x: batch, y: batch, keep_prob: 0.8})

    if not i % 200:
        ls, d, i_ls, d_ls, mu, sigma = sess.run([loss, dec, img_loss, latent_loss, mn, sd],
                                                feed_dict={x: batch, y: batch, keep_prob: 1})
        plt.imshow(np.reshape(batch[0], [28, 28]), cmap='gray')
        plt.show()
        plt.imshow(d[0], cmap='gray')
        plt.show()
        print(i, ls, np.mean(i_ls), np.mean(d_ls))

randoms = [np.random.normal(0, 1, n_latent) for _ in range(10)]
imgs = sess.run(dec, feed_dict = {sampled: randoms, keep_prob: 1.0})
imgs = [np.reshape(imgs[i], [28, 28]) for i in range(len(imgs))]

for img in imgs:
    plt.figure(figsize=(1,1))
    plt.axis('off')
    plt.imshow(img, cmap='gray')

