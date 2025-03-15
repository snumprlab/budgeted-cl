import math
import torch

def centering(K, device):
    n = K.shape[0]
    unit = torch.ones([n, n]).to(device)
    I = torch.eye(n).to(device)
    H = I - unit / n
    return torch.mm(torch.mm(H.float(), K.float()), H.float())  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return torch.mm(H, K)  # KH


def rbf(X, sigma=None):
    GX = torch.mm(X, X.T)
    KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
    if sigma is None:
        mdist = torch.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = torch.exp(KX)
    return KX


def kernel_HSIC(X, Y, sigma):
    return torch.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


def linear_HSIC(X, Y, device):
    L_X = torch.mm(X, X.T)
    L_Y = torch.mm(Y, Y.T)
    return torch.sum(centering(L_X, device) * centering(L_Y, device))


def linear_CKA(X, Y, device):
    hsic = linear_HSIC(X, Y, device)
    var1 = torch.sqrt(linear_HSIC(X, X, device))
    var2 = torch.sqrt(linear_HSIC(Y, Y, device))

    return hsic / (var1 * var2)


def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = torch.sqrt(kernel_HSIC(X, X, sigma))
    var2 = torch.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)


'''
if __name__=='__main__':
    X = torch.randn(100, 64)
    Y = torch.randn(100, 64)

    print('Linear CKA, between X and Y: {}'.format(linear_CKA(X, Y)))
    print('Linear CKA, between X and X: {}'.format(linear_CKA(X, X)))

    print('RBF Kernel CKA, between X and Y: {}'.format(kernel_CKA(X, Y)))
    print('RBF Kernel CKA, between X and X: {}'.format(kernel_CKA(X, X)))
'''