import torch

def chol(A):
    return A.cholesky(upper=False)

def chol_solve(L, y):
    return torch.potrs(y, L, upper = False)

def logDet(L):
    return 2 * L.diag().log().sum()
