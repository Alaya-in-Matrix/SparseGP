import torch
import numpy as np

def chol(A):
    return A.cholesky(upper=False)

def chol_solve(L, y):
    lx, _ = torch.trtrs(y,  L, upper = False);
    x, _  = torch.trtrs(lx, L, upper = False, transpose = True);
    return x.reshape(y.shape)

def logDet(L):
    return 2 * L.diag().log().sum()

def v2tril(v, n):
    """
    Convert a 1D torch tensor to lower triangular matrix
    """
    m = torch.zeros(n, n)
    m[np.tril_indices(n)] = v
    return m

def tril2v(m, n):
    """
    Convert a n*n matrix into a 1D torch tensor
    """
    return m[np.tril_indices(n)]

