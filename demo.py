from VFE import VFE
from sklearn.datasets import load_boston
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from util import *
import torch
import numpy as np
from math import pi

torch.set_default_tensor_type('torch.DoubleTensor')

train_x = torch.tensor(np.loadtxt('./datasets/kin_40k/train_x'))
train_y = torch.tensor(np.loadtxt('./datasets/kin_40k/train_y'))
test_x  = torch.tensor(np.loadtxt('./datasets/kin_40k/test_x'))
test_y  = torch.tensor(np.loadtxt('./datasets/kin_40k/test_y'))

conf                 = dict()
conf['num_inducing'] = 256
conf['debug']        = False
conf['num_epoch']    = 300
conf['jitter_u']     = 1e-6
conf['kmeans']       = True

model = VFE(train_x, train_y, conf)
model.train_scipy()
py, ps2 = model.predict(test_x)

mse  = torch.mean((test_y - py)**2)
smse = mse / torch.var(test_y)

log_prob1 = -0.5 * torch.log(2 * pi * ps2)           - 0.5 * (test_y - py)**2 / ps2
log_prob2 = -0.5 * torch.log(2 * pi * train_y.var()) - 0.5 * (test_y - train_y.mean())**2 / train_y.var()
snlp      = -1 * (log_prob1 - log_prob2).mean()
print('smse = %g' % smse)
print('snlp = %g' % snlp)

np.savetxt('py', py.detach().numpy())
np.savetxt('ps2', ps2.detach().numpy())
np.savetxt('test_y', test_y.detach().numpy())
