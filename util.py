import torch

def chol(A):
    return A.potrf(upper=False)

def chol_solve(L, y):
    lx, _ = torch.trtrs(y,  L, upper = False);
    x, _  = torch.trtrs(lx, L, upper = False, transpose = True);
    return x

def logDet(L):
    return 2 * L.diag().log().sum()
