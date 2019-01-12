from util import *
from math import pi
import torch
import numpy as np
import traceback
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.optimize import minimize, fmin_l_bfgs_b, fmin_cg
import sys

class SVIGP(VFE):
    def __init__(self, train_x, train_y, conf):
        """
        SVI-GP: Gaussian process with mini-batch optimization, as described in Hensman, James, Nicolo Fusi, and Neil D. Lawrence. "Gaussian processes for big data." arXiv preprint arXiv:1309.6835 (2013).
        """
        super(SVIGP, self).__init__(train_x, train_y, conf)
        self.batch_size = conf.get('batch_size', 128)

    def optimal_q(self, n_samples = 200):
        """
        Calculate the analytical optimal q(u) using  `n_samples` data points
        """
        m = torch.zeros(self.m)
        S = torch.eye(self.m)
        return m, S

    def init_hyper(self):
        pass

    def hyper_requires_grad(self, req_grad = True):
        pass

    def loss(self, X, y):
        pass

    def train(self):
        pass

    def post_train(self):
        pass

    def predict(self, x):
        pass
