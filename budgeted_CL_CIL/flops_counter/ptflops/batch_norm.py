import numpy as np

class Batchnorm():

    def __init__(self,X_dim):
        self.d_X, self.h_X, self.w_X = X_dim
        self.gamma = np.ones((1, int(np.prod(X_dim)) ))
        self.beta = np.zeros((1, int(np.prod(X_dim))))
        self.params = [self.gamma, self.beta]

    def forward(self,X):
        '''
        each 제곱 x
        다 더하고 나누는거 x
        
        flops: self.X_flat dimension * 7 : (sub, div, mul, sum) + torch.mean flops (len(x)) + torch.var flops (len(x) + len(x))
        '''
        self.n_X = X.shape[0]
        self.X_shape = X.shape
        
        self.X_flat = X.ravel().reshape(self.n_X,-1)
        self.mu = np.mean(self.X_flat, axis=0)
        self.var = np.var(self.X_flat, axis=0)
        self.X_norm = (self.X_flat - self.mu)/np.sqrt(self.var + 1e-8) # self.X_flat dimension
        out = self.gamma * self.X_norm + self.beta # self.X_flat dimension
        return out.reshape(self.X_shape)

    def backward(self,dout):
        '''
        flops: self.X_flat dimension * 15
        '''
        dout = dout.ravel().reshape(dout.shape[0],-1) 
        X_mu = self.X_flat - self.mu # self.X_flat dimension 1번
        var_inv = 1./np.sqrt(self.var + 1e-8)         
        
        dbeta = np.sum(dout,axis=0) # 1번
        dgamma = dout * self.X_norm # self.X_flat dimension 1번
        dX_norm = dout * self.gamma # self.X_flat dimension 1번
        dvar = np.sum(dX_norm * X_mu,axis=0) * -0.5 * (self.var + 1e-8)**(-3/2) # self.X_flat dimension * 2
        dmu = np.sum(dX_norm * -var_inv ,axis=0) + dvar * 1/self.n_X * np.sum(-2.* X_mu, axis=0) # self.X_flat dimension * 5
        dX = (dX_norm * var_inv) + (dmu / self.n_X) + (dvar * 2/self.n_X * X_mu) # self.X_flat dimension * 4
        dX = dX.reshape(self.X_shape)
        return dX, [dgamma, dbeta]

bn = Batchnorm([16, 32, 32])
X = np.ones([1, 16, 32, 32])
dout = np.ones([1, 16, 32, 32])

out = bn.forward(X)
print(out.shape)

dout_return = bn.backward(dout)