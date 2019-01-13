from util  import *
from SVIGP import SVIGP
from math  import pi
import numpy as np
import torch

torch.set_default_tensor_type('torch.DoubleTensor')
# torch.manual_seed(0)
# np.random.seed(0)

conf                 = dict()
conf['debug']        = True
conf['batch_size']   = 512
conf['num_inducing'] = 256
conf['num_epoch']    = 300
conf['kmeans']       = True
conf['lr']           = 0.001
conf['fix_u']        = False
conf['jitter_u']     = 1e-4
train_x              = torch.tensor(np.loadtxt('./datasets/kin_40k/train_x'))
train_y              = torch.tensor(np.loadtxt('./datasets/kin_40k/train_y'))
m                    = SVIGP(train_x, train_y, conf)
m.train()

test_x  = torch.tensor(np.loadtxt('./datasets/kin_40k/test_x'))
test_y  = torch.tensor(np.loadtxt('./datasets/kin_40k/test_y'))
py, ps2 = m.predict(test_x)
np.savetxt('py', py.detach().numpy())
np.savetxt('ps2', ps2.detach().numpy())


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
