#! /usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

mu = tf.Variable(tf.constant(0.), dtype=tf.float32)
sigma = tf.Variable(tf.constant(1.), dtype=tf.float32)
normal_distributions = tf.contrib.distributions.Normal(mu, sigma)
nd_op = normal_distributions.sample(1)

mu = tf.Variable(tf.constant([0., 100.]), dtype=tf.float32)
sigma = tf.Variable(tf.constant([1., 1.]), dtype=tf.float32)
two_normal_distributions = tf.contrib.distributions.Normal(mu, sigma)
tnd_op = two_normal_distributions.sample(1)

mu = tf.Variable(tf.constant([0., 100.]), dtype=tf.float32)
sigma = tf.Variable(tf.constant([1., 1.]), dtype=tf.float32)
multi_normal_distributions = tf.contrib.distributions.MultivariateNormalDiag(tf.multiply(mu, tf.constant(2.)), sigma)
mnd_op = two_normal_distributions.sample(1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    nd_value, tnd_value, mnd_value = sess.run([nd_op, tnd_op, mnd_op])
    print ('nd_value: ', nd_value)
    print ('tnd_value: ', tnd_value)
    print ('mnd_value: ', mnd_value.shape)

