#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class DataDistribution(object):
    def __init__(self, mu=4, sigma=1):
        self.mu = mu
        self.sigma = sigma

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples


class GeneratorDistribution(object):
    def __init__(self, range):
        self.range = range

    def sample(self, N):
        samples = np.linspace(-self.range, self.range, N) + np.random.random(N)*0.01
        return samples


def generator():
    pass
