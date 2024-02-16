import mytorch as mt

from abc import ABC, abstractmethod

from math import pi, sqrt

class Activation(ABC):
    ''' Base class for all loss functions. '''

    @abstractmethod
    def forward(self, x: mt.Tensor) -> mt.Tensor:
        ''' Forward pass. '''

        pass

    def __call__(self, x: mt.Tensor) -> mt.Tensor:
        ''' When the object is called, it calls the forward method. '''

        return self.forward(x)


class Sigmoid(Activation):
    ''' Sigmoid activation function. '''

    def forward(self, x: mt.Tensor):
        return 1 / (1 + mt.exp(-x))

class ReLU(Activation):
    ''' Rectified Linear Unit activation function. '''

    def forward(self, x: mt.Tensor):
        return mt.maximum(x, 0)

class LeakyReLU(Activation):
    ''' Leaky Rectified Linear Unit activation function. '''

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def forward(self, x: mt.Tensor):
        return mt.where(x > 0, x, self.alpha * x)

class Tanh(Activation):
    ''' Hyperbolic Tangent activation function. '''

    def forward(self, x: mt.Tensor):
        return mt.tanh(x)

class Softplus(Activation):
    ''' Softplus activation function. '''

    def __init__(self, beta: float = 1):
        self.beta = beta

    def forward(self, x: mt.Tensor):
        return (1 / self.beta) * mt.log(1 + mt.exp(self.beta * x))

class Softsign(Activation):
    ''' Softsign activation function. '''

    def forward(self, x: mt.Tensor):
        return x / (1 + mt.abs(x))

class ELU(Activation):
    ''' Exponential Linear Unit activation function. '''

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def forward(self, x: mt.Tensor):
        return mt.where(x > 0, x, self.alpha * (mt.exp(x) - 1))

class SELU(Activation):
    ''' Scaled Exponential Linear Unit activation function. '''

    def __init__(self):
        self.scale = 1.0507009873554804934193349852946
        self.alpha = 1.6732632423543772848170429916717

    def forward(self, x: mt.Tensor):
        return self.scale * mt.where(x > 0, x, self.alpha * (mt.exp(x) - 1))

class GELU(Activation):
    ''' Gaussian Error Linear Unit activation function. '''

    def forward(self, x: mt.Tensor):
        return 0.5 * x * (1 + mt.tanh(sqrt(2 / pi) * (x + 0.044715 * x ** 3)))

class Softmax(Activation):
    ''' Softmax activation function. '''

    def __init__(self, axis: int = 1):
        self.axis = axis

    def forward(self, x: mt.Tensor):
        e_x = mt.exp(x - mt.max(x, axis=self.axis, keepdims=True))

        return e_x / mt.sum(e_x, axis=self.axis, keepdims=True)
