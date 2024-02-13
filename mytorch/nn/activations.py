from abc import ABC, abstractmethod

from mytorch.autograd import Tensor

import numpy as np

from math import pi, sqrt

class Activation(ABC):
    ''' Base class for all loss functions. '''
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        ''' Forward pass. '''
        
        pass
    
    def __call__(self, x: Tensor) -> Tensor:
        ''' When the object is called, it calls the forward method. '''
        
        return self.forward(x)
    
class Sigmoid(Activation):
    ''' Sigmoid activation function. '''
    
    def forward(self, x: Tensor) -> Tensor:
        return 1 / (1 + (-x).exp())
    
class ReLU(Activation):
    ''' Rectified Linear Unit activation function. '''
    
    def forward(self, x: Tensor):
        return x.maximum(0)
    
class LeakyReLU(Activation):
    ''' Leaky Rectified Linear Unit activation function. '''
    
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        
    def forward(self, x: Tensor):
        return (x > 0).where(x, self.alpha * x)
    
class Tanh(Activation):
    ''' Hyperbolic Tangent activation function. '''
    
    def forward(self, x: Tensor):
        return x.tanh()

class Softplus(Activation):
    ''' Softplus activation function. '''
    
    def __init__(self, beta: float = 1):
        self.beta = beta
    
    def forward(self, x: Tensor):
        return (1 / self.beta) * (1 + (self.beta * x).exp()).log()
    
class Softsign(Activation):
    ''' Softsign activation function. '''
    
    def forward(self, x: Tensor):
        return x / (1 + x.abs())
    
class ELU(Activation):
    ''' Exponential Linear Unit activation function. '''
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
         
    def forward(self, x: Tensor):
        return (x > 0).where(x, self.alpha * (x.exp() - 1))
    
class SELU(Activation):
    ''' Scaled Exponential Linear Unit activation function. '''
    
    def __init__(self):
        self.scale = 1.0507009873554804934193349852946
        self.alpha = 1.6732632423543772848170429916717
        
    def forward(self, x: Tensor):
        return self.scale * (x > 0).where(x, self.alpha * (x.exp() - 1))
    
class GELU(Activation):
    ''' Gaussian Error Linear Unit activation function. '''
    
    def forward(self, x: Tensor):
        return 0.5 * x * (1 + (sqrt(2 / pi) * (x + 0.044715 * x ** 3)).tanh())
    
class Softmax(Activation):
    ''' Softmax activation function. '''
    
    def __init__(self, axis: int = 1):
        self.axis = axis
    
    def forward(self, x: Tensor):
        e_x = (x - x.max(axis=self.axis, keepdims=True)).exp()
        
        return e_x / e_x.sum(axis=self.axis, keepdims=True)
