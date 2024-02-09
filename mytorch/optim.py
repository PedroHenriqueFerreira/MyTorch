from abc import ABC, abstractmethod

from mytorch.autograd import Tensor

import numpy as np

class Optimizer(ABC):
    ''' Base class for all optimizers. '''
    
    @abstractmethod
    def step(self):
        ''' Update the parameters. '''
        
        pass
    
    def zero_grad(self):
        ''' Zero the gradients of the parameters. '''
        
        for param in self.params:
            if param.grad is None:
                continue
            
            param.zero_grad()      

class SGD(Optimizer):
    ''' Stochastic Gradient Descent optimizer. '''
    
    def __init__(
        self, 
        params: list[Tensor], 
        lr: float = 1e-2, 
        momentum: float = 0,
        nesterov: bool = False  
    ):
        
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        
        self.m = [np.zeros(param.shape) for param in self.params]
        
    def step(self):
        
        for i, param in enumerate(self.params):
            
            if param.grad is None:
                continue
            
            self.m[i] = self.momentum * self.m[i] + self.lr * param.grad
                
            if self.nesterov:
                param -= self.momentum * self.m[i] + self.lr * param.grad
            else:
                param -= self.m[i]

class Adagrad:
    ''' Adaptive Gradient optimizer. '''
    
    def __init__(
        self, 
        params: list[Tensor], 
        lr: float = 1e-2, 
        eps: float = 1e-8
    ):
        
        self.params = params
        self.lr = lr
        self.eps = eps
        
        self.v = [np.zeros(param.shape) for param in self.params]
        
    def step(self):
            
        for i, param in enumerate(self.params):
            
            if param.grad is None:
                continue
            
            self.v[i] += param.grad ** 2
            
            param -= self.lr * param.grad / (np.sqrt(self.v[i]) + self.eps)

class RMSProp:
    ''' Root Mean Square Propagation optimizer. '''
    
    def __init__(
        self, 
        params: list[Tensor], 
        lr: float = 1e-2, 
        alpha: float = 0.99,
        eps: float = 1e-8
    ):
        
        self.params = params
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        
        self.v = [np.zeros(param.shape) for param in self.params]
    
    def step(self):
        
        for i, param in enumerate(self.params):
            
            if param.grad is None:
                continue
            
            self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * param.grad ** 2
            
            param -= self.lr * param.grad / (np.sqrt(self.v[i]) + self.eps)

class Adadelta:
    ''' Adaptive Delta optimizer. '''
    
    def __init__(
        self, 
        params: list[Tensor], 
        lr: float = 1.0, 
        rho: float = 0.9,
        eps: float = 1e-6
    ):
        
        self.params = params
        self.lr = lr
        self.rho = rho
        self.eps = eps
        
        self.v = [np.zeros(param.shape) for param in self.params]
        self.u = [np.zeros(param.shape) for param in self.params]
        
    def step(self):
        
        for i, param in enumerate(self.params):
            
            if param.grad is None:
                continue
            
            self.v[i] = self.rho * self.v[i] + (1 - self.rho) * param.grad ** 2
            delta = np.sqrt(self.u[i] + self.eps) / np.sqrt(self.v[i] + self.eps) * param.grad
            self.u[i] = self.rho * self.u[i] + (1 - self.rho) * delta ** 2
            
            param -= self.lr * delta

class Adam(Optimizer):
    ''' Adaptive Moment Estimation optimizer. '''
    
    def __init__(
        self, 
        params: list[Tensor], 
        lr: float = 1e-2, 
        betas: tuple[float, float] = (0.9, 0.999), 
        eps: float = 1e-8
    ):
        
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        
        self.m = [np.zeros(param.shape) for param in self.params]
        self.v = [np.zeros(param.shape) for param in self.params]
        
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
    
class Adamax(Optimizer):
    ''' Adam with infinity norm. '''
    
    def __init__(
        self, 
        params: list[Tensor], 
        lr: float = 2e-3, 
        betas: tuple[float, float] = (0.9, 0.999), 
        eps: float = 1e-8
    ):
        
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        
        self.m = [np.zeros(param.shape) for param in self.params]
        self.u = [np.zeros(param.shape) for param in self.params]
        
        self.t = 0
        
    def step(self):
        
        self.t += 1
        
        for i, param in enumerate(self.params):
            
            if param.grad is None:
                continue
            
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * param.grad
            self.u[i] = np.maximum(self.betas[1] * self.u[i], np.abs(param.grad))
            
            m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
            
            param -= self.lr * m_hat / (self.u[i] + self.eps)