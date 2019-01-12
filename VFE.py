from util            import *
from math            import pi
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.utils   import shuffle
from scipy.optimize  import fmin_l_bfgs_b
import torch
import numpy as np
import traceback
import sys

class VFE:

    # TODO: specify inducing points from constructor
    def __init__(self, train_x, train_y, conf):
        self.m                = min(train_x.shape[0], conf.get('num_inducing', 200))
        self.debug            = conf.get('debug', False)
        self.num_epoch        = conf.get('num_epoch', 200)
        self.kmeans           = conf.get('kmeans', True)
        self.lr               = conf.get('lr', 0.005)
        self.rv               = conf.get('rv', 1.0)
        self.rl               = conf.get('rl', 1.0)
        self.jitter_u         = conf.get('jitter_u', 1e-6)
        self.num_train        = train_x.shape[0]
        self.dim              = train_x.shape[1]
        self.x                = train_x.clone()
        self.y                = train_y.clone()
        self.xmean, self.xstd = self.x.mean(dim=0), self.x.std(dim=0)
        self.ymean, self.ystd = self.y.mean(dim=0), self.y.std(dim=0)
        self.x                = (self.x - self.xmean) / self.xstd
        self.y                = (self.y - self.ymean) / self.ystd
    
    # TODO: refer to the pytorch multi-variate normal distribution to speedup this function
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
        dist    = (x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))).clamp(min = 0)
        return sf2 * torch.exp(-0.5 * dist)
        # dist    = torch.zeros(num_x1, num_x2)
        # for i in range(num_x1):
        #     dist[i] = (x[i] - y).pow(2).sum(dim = 1)

    def init_hyper(self):
        self.log_sf      = torch.log(torch.tensor(self.rv)).double();
        self.log_sn      = torch.log(torch.tensor(1e-3)).double();
        self.log_lscales = torch.log(self.rl * torch.ones(self.dim)).double();
        if self.kmeans:
            self.u = self.kmeans_init()
        else:
            self.u = torch.tensor(shuffle(self.x.numpy(), n_samples = self.m))
    
    def hyper_requires_grad(self, req_grad = True):
        self.log_sf.requires_grad      = req_grad
        self.log_sn.requires_grad      = req_grad
        self.log_lscales.requires_grad = req_grad
        self.u.requires_grad           = req_grad

    def kmeans_init(self):
        """
        Use K-means to initialize the inducing points
        """
        return torch.tensor(MiniBatchKMeans(n_clusters = self.m).fit(self.x).cluster_centers_)

    def loss(self, X, y):
        """
        X: num_x * dim
        y: num_x vector
        """
        num_x  = X.shape[0]
        sf2    = torch.exp(2 * self.log_sf)
        sn2    = torch.exp(2 * self.log_sn)
        Kuu    = self.cov(self.u, self.u) + self.jitter_u * torch.eye(self.m)
        Kux    = self.cov(self.u, X)
        Kxu    = Kux.t()
        Luu    = chol(Kuu)
        A      = sn2 * Kuu + Kux.mm(Kxu)
        LA     = chol(A)
        Kuxy   = Kux.mv(y)

        # -0.5 * (y' *  inv(Q + sn2 * I) * y)
        loss_1    = - 0.5 * (y.dot(y) - Kuxy.dot(chol_solve(LA, Kuxy.unsqueeze(1)).squeeze())) / sn2

        # -0.5 * (log |Q + sn2 * I| + num_x * log(2 * pi))
        log_det_K = (num_x - self.m) * torch.log(sn2) + logDet(LA) - logDet(Luu)
        loss_2    = -0.5 * (log_det_K + num_x * torch.log(torch.tensor(2 * pi)))
       
        # -(0.5 / sn2) * Trace(K - Q)
        loss_3    = -0.5 * (sf2 * num_x - torch.sum(Kxu * chol_solve(Luu, Kux).t())) / sn2

        # loss = log(N(y | 0, sn2 * eye(num_x) + Q)) - (0.5/sn2) * Trace(K - Q)
        loss      = loss_1 + loss_2 + loss_3
        
        return -1 * loss

    def train(self):
        self.init_hyper()
        self.hyper_requires_grad(True)
        # opt = torch.optim.Adam([self.log_sf, self.log_lscales, self.log_sn, self.u], lr = self.lr) # TODO: lr scheduler
        opt1 = torch.optim.LBFGS([self.log_sf, self.log_lscales, self.log_sn, self.u], max_iter = 10)
        opt2 = torch.optim.Adam([self.log_sf, self.log_lscales, self.log_sn, self.u], lr = self.lr)
        try:
            for step in range(5):
                def closure():
                    opt1.zero_grad()
                    loss = self.loss(self.x, self.y)
                    loss.backward()
                    return loss
                opt1.step(closure)
                print('BFGS Epoch %d, loss = %g' % (step, self.loss(self.x, self.y)))
            for step in range(self.num_epoch):
                def closure():
                    opt2.zero_grad()
                    loss = self.loss(self.x, self.y)
                    loss.backward()
                    return loss
                opt2.step(closure)
                print('Epoch %d, loss = %g' % (step, self.loss(self.x, self.y)))
        except RuntimeError:
            if self.debug:
                print("Failed to perform Cholesky decomposition, stop optimization")
        self.post_train()

    def train_scipy(self):
        self.init_hyper()
        self.hyper_requires_grad(True)
        def obj(x):
            self.log_sf      = torch.tensor(x[0])
            self.log_sn      = torch.tensor(x[1])
            self.log_lscales = torch.tensor(x[2:2 + self.dim])
            self.u           = torch.tensor(x[2+self.dim:]).reshape(self.m, self.dim)
            self.hyper_requires_grad(True)
            loss             = self.loss(self.x, self.y)
            loss.backward()
            grad               = np.zeros(x.shape)
            grad[0]            = self.log_sf.grad.detach().numpy()
            grad[1]            = self.log_sn.grad.detach().numpy()
            grad[2:2+self.dim] = self.log_lscales.grad.detach().numpy()
            grad[2+self.dim:]   = self.u.grad.reshape(self.m * self.dim).detach().numpy()
            return loss, grad
        x0               = np.zeros(2 + self.dim + self.dim * self.m)
        x0[0]            = self.log_sf.detach().numpy()
        x0[1]            = self.log_sn.detach().numpy()
        x0[2:2+self.dim] = self.log_lscales.detach().numpy()
        x0[2+self.dim:]  = self.u.reshape(self.m * self.dim).detach().numpy()
        ox, acq_f, _     = fmin_l_bfgs_b(obj, x0, maxiter = self.num_epoch, m = 10, iprint=1)
        self.log_sf      = torch.tensor(ox[0])
        self.log_sn      = torch.tensor(ox[1])
        self.log_lscales = torch.tensor(ox[2:2 + self.dim])
        self.u           = torch.tensor(ox[2+self.dim:]).reshape(self.m, self.dim)
        self.post_train()

    def post_train(self):
        self.hyper_requires_grad(False)
        sn2        = torch.exp(2 * self.log_sn)
        Kuu        = self.cov(self.u, self.u) + self.jitter_u * torch.eye(self.m)
        Kux        = self.cov(self.u, self.x)
        Kxu        = Kux.t()
        Luu        = chol(Kuu)
        S          = Kuu + Kux.mm(Kxu) / sn2
        LS         = chol(S)
        self.sf2   = torch.exp(2 * self.log_sf)
        self.sn2   = sn2
        self.mu    = Kuu.mv(chol_solve(LS, Kux.mv(self.y))) / sn2
        self.A     = Kuu.mm(chol_solve(LS, Kuu))
        self.Luu   = Luu
        self.alpha = chol_solve(Luu, self.mu)
        
    def predict(self, x):
        tx = x;
        if tx.dim() == 1:
            tx = tx.reshape(1, tx.numel())
        tx         = (tx - self.xmean) / self.xstd
        Kxu        = self.cov(tx, self.u)
        Kux        = Kxu.t()
        invKuu_Kux = chol_solve(self.Luu, Kux)
        py         = self.ymean + self.ystd * Kxu.mv(self.alpha)
        ps2        = self.ystd**2 * (self.sf2 + self.sn2 - ((Kxu * invKuu_Kux.t()) + (invKuu_Kux * self.A.mm(invKuu_Kux)).t()).sum(dim = 1))
        return py, ps2.clamp(min = self.ystd**2 * self.sn2)

    def BO_obj(self):
        pass

    def plot_1d(self):
        pass
