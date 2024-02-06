from autograd import Tensor

import numpy as np

class Optimizer:
    ''' Base class for all optimizers. '''
    
    def __init__(self, params: list[Tensor]):
        ''' Initializing the optimizer parameters. '''
        self.params = params
    
    def step(self):
        ''' Update the parameters. '''
        
        raise NotImplementedError()
    
    def zero_grad(self):
        ''' Zero the gradients of the parameters. '''
        
        for param in self.params:
            if param.grad is None:
                continue
            
            param.zero_grad()
                
class Adam(Optimizer):
    ''' Adaptive Moment Estimation optimizer. '''
    
    def __init__(
        self, 
        params: list[Tensor], 
        lr: float = 1e-2, 
        betas: tuple[float, float] = (0.9, 0.999), 
        eps: float = 1e-8
    ):
        
        super().__init__(params)
        
        self.lr = lr
        self.betas = betas
        self.eps = eps
        
        self.m = [np.zeros_like(param) for param in self.params]
        self.v = [np.zeros_like(param) for param in self.params]
        
        self.t = 0
        
    def step(self):
        
        self.t += 1
        
        for i, param in enumerate(self.params):
            
            if param.grad is None:
                continue
            
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * param.grad
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * param.grad ** 2
            
            m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
            v_hat = self.v[i] / (1 - self.betas[1] ** self.t)
            
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class SGD(Optimizer):
    ''' Stochastic Gradient Descent optimizer. '''
    
    