from abc import ABC, abstractmethod

from mytorch.autograd import Tensor

import numpy as np

class Activation(ABC):
    ''' Base class for all loss functions. '''
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        ''' Forward pass. '''
        
        pass
    
    def __call__(self, x: Tensor) -> Tensor:
        ''' When the object is called, it calls the forward method. '''
        
        return self.forward(x)
    
class Sigmoid:
    ''' Sigmoid activation function. '''
    
    def forward(self, x: Tensor) -> Tensor:
        data = 1 / (1 + np.exp(-x.data))
        requires_grad = x.requires_grad
        grad_fn = None
        
        if requires_grad:
            def grad_fn(grad: np.ndarray):
                x.backward(grad * data * (1 - data))
        
        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)
    
class ReLU(Activation):
    ''' Rectified Linear Unit activation function. '''
    
    def forward(self, x: Tensor):
        data = np.maximum(0, x.data)
        requires_grad = x.requires_grad
        grad_fn = None
        
        if requires_grad:
            def grad_fn(grad: np.ndarray):
                x.backward(grad * (data > 0))
                
        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)
    
class LeakyReLU(Activation):
    ''' Leaky Rectified Linear Unit activation function. '''
    
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        
    def forward(self, x: Tensor):
        data = np.where(x.data <= 0, x.data * self.alpha, x.data)
        requires_grad = x.requires_grad
        grad_fn = None
        
        if requires_grad:
            def grad_fn(grad: np.ndarray):
                x.backward(grad * np.where(data <= 0, self.alpha, 1))
                
        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)
    
class Tanh(Activation):
    ''' Hyperbolic Tangent activation function. '''
    
    def forward(self, x: Tensor):
        data = np.tanh(x.data)
        requires_grad = x.requires_grad
        grad_fn = None
        
        if requires_grad:
            def grad_fn(grad: np.ndarray):
                x.backward(grad * (1 - data ** 2))
                
        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

class Softplus(Activation):
    ''' Softplus activation function. '''
    
    def forward(self, x: Tensor):
        data = np.log(1 + np.exp(x.data))
        requires_grad = x.requires_grad
        grad_fn = None
        
        if requires_grad:
            def grad_fn(grad: np.ndarray):
                x.backward(grad * 1 / (1 + np.exp(-x.data)))
                
        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)
    
class Softsign(Activation):
    ''' Softsign activation function. '''
    
    def forward(self, x: Tensor):
        data = x.data / (1 + np.abs(x.data))
        requires_grad = x.requires_grad
        grad_fn = None
        
        if requires_grad:
            def grad_fn(grad: np.ndarray):
                x.backward(grad / (1 + np.abs(x.data)) ** 2)
                
        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)
    
class ELU(Activation):
    ''' Exponential Linear Unit activation function. '''
    
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        
    def forward(self, x: Tensor):
        data = np.where(x.data <= 0, self.alpha * (np.exp(x.data) - 1), x.data)
        requires_grad = x.requires_grad
        grad_fn = None
        
        if requires_grad:
            def grad_fn(grad: np.ndarray):
                x.backward(grad * np.where(data <= 0, data + self.alpha, 1))
                
        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)
    
class SELU(Activation):
    ''' Scaled Exponential Linear Unit activation function. '''
    
    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.lmbda = 1.0507009873554804934193349852946
        
    def forward(self, x: Tensor):
        data = self.lmbda * np.where(x.data <= 0, self.alpha * (np.exp(x.data) - 1), x.data)
        requires_grad = x.requires_grad
        grad_fn = None
        
        if requires_grad:
            def grad_fn(grad: np.ndarray):
                x.backward(grad * self.lmbda * np.where(data <= 0, self.alpha * np.exp(x.data), 1))
                
        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)
    
class GELU(Activation): # TODO: Check gradient
    ''' Gaussian Error Linear Unit activation function. '''
    
    def forward(self, x: Tensor):
        data = 0.5 * x.data * (1 + np.tanh(np.sqrt(2 / np.pi) * (x.data + 0.044715 * x.data ** 3)))
        requires_grad = x.requires_grad
        grad_fn = None
        
        if requires_grad:
            def grad_fn(grad: np.ndarray):
                x.backward(grad * (
                    0.5 * np.tanh(0.0356774 * x.data ** 3 + 0.797885 * x.data) 
                    + (0.0535161 * x.data ** 3 + 0.398942 * x.data)
                    * (1 / np.cosh(0.0356774 * x.data ** 3 + 0.797885 * x.data)) ** 2
                    + 0.5
                ))
                
        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)
    
class Softmax(Activation):
    ''' Softmax activation function. '''
    
    def __init__(self, axis: int = 1):
        self.axis = axis
    
    def forward(self, x: Tensor):
        e_x = np.exp(x.data - np.max(x.data, axis=self.axis, keepdims=True))
        data = e_x / np.sum(e_x, axis=self.axis, keepdims=True)
        requires_grad = x.requires_grad
        grad_fn = None
        
        if requires_grad:
            def grad_fn(grad: np.ndarray):
                x.backward(grad * data * (1 - data))
                
        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)
    
class Softmax2(Activation): # TODO: Check gradient
    ''' Softmax activation function. '''
    
    def __init__(self, axis: int = 1):
        self.axis = axis
    
    def forward(self, x: Tensor):
        e_x = (x - x.max(axis=self.axis, keepdims=True)).exp()
        
        return e_x / e_x.sum(axis=self.axis, keepdims=True)