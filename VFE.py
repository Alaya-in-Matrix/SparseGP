from util import *
from math import pi
import torch

class VFE:

    # TODO: specify inducing points from constructor
    def __init__(self, train_x, train_y, conf):
        self.m                = conf.get('num_inducing', 200)
        self.debug            = conf.get('debug', False)
        self.jitter_u         = 1e-15
        self.num_train        = train_x.shape[0]
        self.dim              = train_x.shape[1]
        self.x                = train_x.clone()
        self.y                = train_y.clone()
        self.xmean, self.xstd = self.x.mean(dim=0), self.x.std(dim=0)
        self.ymean, self.ystd = self.y.mean(dim=0), self.y.std(dim=0)
        self.x                = (self.x - self.xmean) / self.xstd
        self.y                = (self.y - self.ymean) / self.ystd

    def cov(self, X1, X2):
        """
        SE ARD kernel
        """
        num_x1  = X1.shape[0]
        num_x2  = X2.shape[0]
        sf2     = torch.exp(2 * self.log_sf)
        sn2     = torch.exp(2 * self.log_sn)
        lscales = torch.exp(self.log_lscales)
        x       = X1 / lscales
        y       = X2 / lscales
        x_norm  = (x**2).sum(1).view(-1, 1) # TODO: understand this line of code
        y_norm  = (y**2).sum(1).view(1, -1)
        dist    = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
        return sf2 * torch.exp(-0.5 * dist);

    def init_hyper(self, rv = 1.0, rl = 1.0):
        self.log_sf                    = torch.log(torch.tensor(rv));
        self.log_sn                    = torch.log(torch.tensor(1e-3));
        self.log_lscales               = torch.log(rl * torch.ones(self.dim));
        self.u                         = torch.randn(self.m, self.dim)
        self.log_sf.requires_grad      = True
        self.log_sn.requires_grad      = True
        self.log_lscales.requires_grad = True
        self.u.requires_grad           = True

    def kmeans_init(self):
        """
        Use K-means to initialize the inducing points
        """
        pass

    def loss(self, X, y):
        """
        X: num_x * dim
        y: num_x vector
        """
        sf2    = torch.exp(2 * self.log_sf)
        sn2    = torch.exp(2 * self.log_sn)
        Kuu    = self.cov(self.u, self.u) + self.jitter_u * torch.eye
        Kxu    = self.cov(X, self.u)
        Kux    = Kxu.t()
        Luu    = chol(Kuu)
        A      = sn2 * Kuu + Kux.mm(Kxu)
        LA     = chol(A)
        Kuxy   = Kux.mv(y)

        # -0.5 * (y^T (Q + sn2 * I)^-1 y)
        loss_1    = - 0.5 * (y.dot(y) - Kuxy.dot(chol_solve(LA, Kuxy))) / sn2

        # -0.5 * (log |Q + sn2 * I| + num_x * log(2 * pi))
        log_det_K = (self.num_x - self.m) * torch.log(sn2) + logDet(LA) - logDet(Luu)
        loss_2    = -0.5 * (log_det_K + self.num_x * torch.log(torch.tensor(2 * pi)))
       
        # -(0.5 / sn2) * Tr(K - Q)
        loss_3 = -0.5 * (sf2 * self.num_x - torch.sum(Kxu * chol_solve(LA, Kux).t())) / sn2

        return loss_1 + loss_2 + loss_3

    def train(self):
        pass

    def predict(self, x):
        py  = self.ymean
        ps2 = self.ystd**2
        return py, ps2

    def BO_obj(self):
        pass

    def plot_1d(self):
        pass