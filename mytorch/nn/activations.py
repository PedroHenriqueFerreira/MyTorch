from mytorch import Tensor
import mytorch as mt

import numpy as np

from abc import ABC, abstractmethod

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

    def forward(self, x: Tensor):
        data = 1 / (1 + np.exp(-x.data))
        requires_grad = x.requires_grad
        sigmoid_backward = None
        
        if requires_grad:
            def sigmoid_backward(grad: np.ndarray):
                x.backward(grad * data * (1 - data))
            
        return Tensor(data, None, requires_grad, sigmoid_backward)

class LogSigmoid(Activation):
    ''' Logarithm of the Sigmoid activation function. '''

    def __init__(self):
        self.softplus = Softplus()

    def forward(self, x: Tensor):
        return -self.softplus(-x)

class ReLU(Activation):
    ''' Rectified Linear Unit activation function. '''

    def forward(self, x: Tensor):
        data = np.where(x.data > 0, x.data, 0)
        requires_grad = x.requires_grad
        relu_backward = None
        
        if requires_grad:
            def relu_backward(grad: np.ndarray):
                x.backward(grad * (x.data > 0))
        
        return Tensor(data, None, requires_grad, relu_backward)

class LeakyReLU(Activation):
    ''' Leaky Rectified Linear Unit activation function. '''

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def forward(self, x: Tensor):
        data = np.where(x.data > 0, x.data, self.alpha * x.data)
        requires_grad = x.requires_grad
        leaky_relu_backward = None

        if requires_grad:
            def leaky_relu_backward(grad: np.ndarray):
                x.backward(grad * np.where(data > 0, 1, self.alpha))
                
        return Tensor(data, None, requires_grad, leaky_relu_backward)

class Tanh(Activation):
    ''' Hyperbolic Tangent activation function. '''

    def forward(self, x: Tensor):
        data = np.tanh(x.data)
        requires_grad = x.requires_grad
        tanh_backward = None
        
        if requires_grad:
            def tanh_backward(grad: np.ndarray):
                x.backward(grad * (1 - data ** 2))
                
        return Tensor(data, None, requires_grad, tanh_backward)

class Softplus(Activation):
    ''' Softplus activation function. '''

    def __init__(self, beta: float = 1, threshold: float = 20):
        self.beta = beta
        self.threshold = threshold

    def forward(self, x: Tensor):
        condition = x.data * self.beta <= self.threshold
        
        data = np.where(condition, (1 / self.beta) * np.log(1 + np.exp(self.beta * x.data)), x.data)
        
        requires_grad = x.requires_grad
        softplus_backward = None
        
        if requires_grad:
            def softplus_backward(grad: np.ndarray):
                x.backward(grad / np.where(condition, 1 + np.exp(-self.beta * x.data), 1))    

        return Tensor(data, None, requires_grad, softplus_backward)

class Softsign(Activation):
    ''' Softsign activation function. '''

    def forward(self, x: Tensor):
        data = x.data / (1 + np.abs(x.data))
        requires_grad = x.requires_grad
        softsign_backward = None
        
        if requires_grad:
            def softsign_backward(grad: np.ndarray):
                x.backward(grad / (1 + np.abs(x.data)) ** 2)
        
        return Tensor(data, None, requires_grad, softsign_backward)

class SiLU(Activation):
    ''' Sigmoid Linear Unit activation function. '''
    
    def __init__(self):
        self.sigmoid = Sigmoid()
    
    def forward(self, x: Tensor):
        return x * self.sigmoid(x)

class Mish(Activation):
    ''' Mish activation function. '''

    def __init__(self):
        self.softplus = Softplus()
        self.tanh = Tanh()

    def forward(self, x: Tensor):
        return x * self.tanh(self.softplus(x))

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
        return self.scale * (x > 0).where(x, self.alpha * (mt.exp(x) - 1))

class GELU(Activation):
    ''' Gaussian Error Linear Unit activation function. '''

    def forward(self, x: Tensor):
        return 0.5 * x * (1 + (sqrt(2 / pi) * (x + 0.044715 * x ** 3)).tanh())

class Softmax(Activation):
    ''' Softmax activation function. '''

    def __init__(self, axis: int = 1):
        self.axis = axis

    def forward(self, x: Tensor):
        e_x = mt.exp(x - x.max(dim=self.axis, keepdim=True))

        return e_x / e_x.sum(dim=self.axis, keepdim=True)
