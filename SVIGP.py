from util             import *
from sklearn.cluster  import KMeans, MiniBatchKMeans
from sklearn.utils    import shuffle
from VFE              import VFE
from math             import pi
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
import sys
import traceback

class SVIGP(VFE):
    """
    SVI-GP: Gaussian process with mini-batch optimization, as described in Hensman, James, Nicolo Fusi, and Neil D. Lawrence. "Gaussian processes for big data." arXiv preprint arXiv:1309.6835 (2013).
    """
    def __init__(self, train_x, train_y, conf):
        super(SVIGP, self).__init__(train_x, train_y, conf)
        self.batch_size = conf.get('batch_size', 128)
        self.fix_u      = conf.get('fix_u', False)
        self.lr_cov     = conf.get('lr_cov', self.lr) # learning rate for log_sf/log_sn/log_lscales
        self.lr_u       = conf.get('lr_u', self.lr)   # learning rate for the locations of inducing points
        self.lr_qm      = conf.get('lr_qm', self.lr)  # learning rate for the mean of q(u)
        self.lr_qL      = conf.get('lr_qL', self.lr)  # learning rate for the covariance matrix of q(u)
        self.bfgs_epoch = conf.get('bfgs_epoch', 1)

    def optimal_q(self, n_samples = 1000):
        """
        Calculate the analytical optimal q(u) using  `n_samples` data points
        """
        n_samples = min(self.num_train, n_samples)
        idxs      = torch.randperm(self.num_train)[:n_samples]
        sn2       = torch.exp(2 * self.log_sn)
        Kuu       = self.cov(self.u, self.u) + self.jitter_u * torch.eye(self.m)
        Kux       = self.cov(self.u, self.x[idxs])
        Kxu       = Kux.t()
        Luu       = chol(Kuu)
        invSigma  = Kuu + Kux.mm(Kxu) / sn2
        LinvSigma = chol(invSigma)
        S         = Kuu.mm(chol_solve(LinvSigma, Kuu))
        m         = Kuu.mv(chol_solve(LinvSigma, Kux.mv(self.y[idxs]))) / sn2
        return m, S

    def init_hyper(self):
        super(SVIGP, self).init_hyper()
        m, S    = self.optimal_q()
        self.qm = m
        self.qL = S.tril()

    def hyper_requires_grad(self, req_grad = True):
        super(SVIGP, self).hyper_requires_grad(req_grad)
        self.qm.requires_grad = req_grad
        self.qL.requires_grad = req_grad
        if self.fix_u:
            self.u.requires_grad = False

    def loss(self, X, y):
        num_x      = X.shape[0]
        sn2        = torch.exp(2 * self.log_sn)
        sf2        = torch.exp(2 * self.log_sf)
        Kuu        = self.cov(self.u, self.u)              + self.jitter_u * torch.eye(self.m)
        S          = self.qL.tril().mm(self.qL.tril().t()) + self.jitter_u * torch.eye(self.m)
        Kux        = self.cov(self.u, X)
        Kxu        = Kux.t()
        Luu        = chol(Kuu)
        LS         = chol(S) # XXX: sometimes diagonal of self.qL would go negative
        invKuu_Kux = chol_solve(Luu, Kux)
        invKuu_m   = chol_solve(Luu, self.qm.unsqueeze(1)).squeeze()

        mu   = Kxu.mv(invKuu_m)
        K_ii = sf2 - (Kxu * invKuu_Kux.t()).sum(dim = 1)

        loss_1 = -0.5 * num_x * torch.log(2 * pi * sn2) - 0.5 * (y - mu).dot(y - mu) / sn2
        loss_2 = -0.5 * K_ii.sum() / sn2
        loss_3 = -0.5 * (invKuu_Kux.t() * S.mm(invKuu_Kux).t()).sum() / sn2
        loss_4 = -0.5 * (chol_solve(Luu, S).trace() + self.qm.dot(invKuu_m) - self.m - logDet(LS) + logDet(Luu)) # KL(q(u) || p(u))
        loss   = self.num_train * (loss_1 + loss_2 + loss_3) / num_x + loss_4
        return -1 * loss

    def set_optimizer(self):
        if self.fix_u:
            self.opt_bfgs = torch.optim.LBFGS([self.log_sf, self.log_sn, self.log_lscales, self.qm, self.qL], history_size = 10, max_iter = 1, lr = 1)
            self.opt = torch.optim.Adam([
                {'params' : [self.log_sf, self.log_sn, self.log_lscales], 'lr': self.lr_cov}, 
                {'params' : [self.qm],                                    'lr': self.lr_qm}, 
                {'params' : [self.qL],                                    'lr': self.lr_qL}])
        else:
            self.opt_bfgs = torch.optim.LBFGS([self.log_sf, self.log_sn, self.log_lscales, self.u, self.qm, self.qL], history_size = 10, max_iter = 1, lr = 1)
            self.opt = torch.optim.Adam([
                {'params' : [self.log_sf, self.log_sn, self.log_lscales], 'lr': self.lr_cov}, 
                {'params' : [self.u],                                     'lr': self.lr_u}, 
                {'params' : [self.qm],                                    'lr': self.lr_qm}, 
                {'params' : [self.qL],                                    'lr': self.lr_qL}])

    def train(self):
        self.init_hyper()
        self.hyper_requires_grad(True)
        self.set_optimizer()

        try:
            dataset = TensorDataset(self.x, self.y)
            loader  = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)
            for epoch in range(self.bfgs_epoch):
                for x, y in loader:
                    def closure():
                        self.opt_bfgs.zero_grad()
                        loss = self.loss(x, y)
                        loss.backward()
                        print('\t%g' % loss, flush = True)
                        return loss
                    self.opt_bfgs.step(closure)
                if self.debug:
                    print("LBFGS Epoch %d" % epoch)
            for epoch in range(self.num_epoch):
                for x, y in loader:
                    self.opt.zero_grad()
                    loss = self.loss(x, y)
                    loss.backward()
                    print('\t%g' % loss, flush = True)
                    self.opt.step()
                if self.debug:
                    print("Adam Epoch %d, loss = %g" % (epoch, loss), flush = True)
        except RuntimeError:
            if self.debug:
                print(traceback.format_exc())
                print('sn2 = %g' % torch.exp(2 * self.log_sn))
                print('sf2 = %g' % torch.exp(2 * self.log_sf))
                np.savetxt('log_lscales', self.log_lscales.detach().numpy())
                np.savetxt('u', self.u.detach().numpy())
                sys.exit(1)
        print("Finished SVI-GP training")
        self.post_train()

    def post_train(self):
        if self.num_train < 100000:
            super(SVIGP, self).post_train() # use the analytical optimal q(u)
        else:
            self.hyper_requires_grad(False)
            sn2        = torch.exp(2 * self.log_sn)
            Kuu        = self.cov(self.u, self.u) + self.jitter_u * torch.eye(self.m)
            Kux        = self.cov(self.u, self.x)
            Kxu        = Kux.t()
            Luu        = chol(Kuu)
            LA         = self.qL.tril()
            self.sf2   = torch.exp(2 * self.log_sf)
            self.sn2   = sn2
            self.mu    = self.qm
            self.A     = LA.mm(LA.t()) + self.jitter_u * torch.eye(self.m)
            self.Luu   = Luu
            self.alpha = chol_solve(Luu, self.mu)


