from mytorch import NDArray, Tensor
from mytorch.nn.modules import Module

class ELU(Module):
    ''' Exponential Linear Unit activation function. '''

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def forward(self, x: Tensor):
        data = x.lib.where(x.data > 0, x.data, self.alpha * (x.lib.exp(x.data) - 1))
        elu_backward = None
        
        if x.requires_grad:
            def elu_backward(grad: NDArray):
                x.backward(grad * x.lib.where(data > 0, 1, self.alpha * x.lib.exp(x.data)))
        
        return Tensor(data, x.dtype, x.requires_grad, elu_backward, x.device)

class LeakyReLU(Module):
    ''' Leaky Rectified Linear Unit activation function. '''

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def forward(self, x: Tensor):
        data = x.lib.where(x.data > 0, x.data, self.alpha * x.data)
        leaky_relu_backward = None

        if x.requires_grad:
            def leaky_relu_backward(grad: NDArray):
                x.backward(grad * x.lib.where(data > 0, 1, self.alpha))
                
        return Tensor(data, x.dtype, x.requires_grad, leaky_relu_backward, x.device)

class ReLU(Module):
    ''' Rectified Linear Unit activation function. '''

    def forward(self, x: Tensor):
        data = x.lib.where(x.data > 0, x.data, 0)
        relu_backward = None
        
        if x.requires_grad:
            def relu_backward(grad: NDArray):
                x.backward(grad * (x.data > 0))
        
        return Tensor(data, x.dtype, x.requires_grad, relu_backward, x.device)

class SELU(Module):
    ''' Scaled Exponential Linear Unit activation function. '''

    def __init__(self):
        self.scale = 1.0507009873554804934193349852946
        self.alpha = 1.6732632423543772848170429916717

    def forward(self, x: Tensor):
        data = self.scale * x.lib.where(x.data > 0, x.data, self.alpha * (x.lib.exp(x.data) - 1))
        selu_backward = None
        
        if x.requires_grad:
            def selu_backward(grad: NDArray):
                x.backward(grad * self.scale * x.lib.where(data > 0, 1, self.alpha * x.lib.exp(x.data)))

        return Tensor(data, x.dtype, x.requires_grad, selu_backward, x.device)

class GELU(Module):
    ''' Gaussian Error Linear Unit activation function. '''

    def forward(self, x: Tensor):
        
        data = 0.5 * x.data * (1 + x.lib.tanh(x.lib.sqrt(2 / x.lib.pi) * (x.data + 0.044715 * x.data ** 3)))
        gelu_backward = None
        
        if x.requires_grad:   
            def gelu_backward(grad: NDArray):
                x.backward(
                    grad * (
                        0.5 * x.lib.tanh(0.0356774 * x.data ** 3 + 0.797885 * x.data) 
                        + (0.0535161 * x.data ** 3 + 0.398942 * x.data) 
                        * (1 / x.lib.cosh(0.0356774 * x.data ** 3 + 0.797885 * x.data)) ** 2
                        + 0.5
                    )
                )
        
        return Tensor(data, x.dtype, x.requires_grad, gelu_backward, x.device)

class Softplus(Module):
    ''' Softplus activation function. '''

    def __init__(self, beta: float = 1, threshold: float = 20):
        self.beta = beta
        self.threshold = threshold

    def forward(self, x: Tensor):
        condition = x.data * self.beta <= self.threshold
        
        data = x.lib.where(condition, (1 / self.beta) * x.lib.log(1 + x.lib.exp(self.beta * x.data)), x.data)
        softplus_backward = None
        
        if x.requires_grad:
            def softplus_backward(grad: NDArray):
                x.backward(grad / x.lib.where(condition, 1 + x.lib.exp(-self.beta * x.data), 1))    

        return Tensor(data, x.dtype, x.requires_grad, softplus_backward, x.device)

class Softsign(Module):
    ''' Softsign activation function. '''

    def forward(self, x: Tensor):
        data = x.data / (1 + x.lib.abs(x.data))
        softsign_backward = None
        
        if x.requires_grad:
            def softsign_backward(grad: NDArray):
                x.backward(grad / (1 + x.lib.abs(x.data)) ** 2)
        
        return Tensor(data, x.dtype, x.requires_grad, softsign_backward, x.device)

class Softmax(Module):
    ''' Softmax activation function. '''

    def __init__(self, dim: int = 1):
        self.dim = dim

    def forward(self, x: Tensor):
        e_x = (x - x.max(dim=self.dim, keepdim=True)).exp()
        return e_x / e_x.sum(dim=self.dim, keepdim=True)


class Sigmoid(Module):
    ''' Sigmoid activation function. '''

    def forward(self, x: Tensor):
        return x.sigmoid()

class Tanh(Module):
    ''' Hyperbolic Tangent activation function. '''

    def forward(self, x: Tensor):
        return x.tanh()

class SiLU(Module):
    ''' Sigmoid Linear Unit activation function. '''
    
    def __init__(self):
        self.sigmoid = Sigmoid()
    
    def forward(self, x: Tensor):
        return x * self.sigmoid(x)

class LogSigmoid(Module):
    ''' Logarithm of the Sigmoid activation function. '''

    def __init__(self):
        self.softplus = Softplus()

    def forward(self, x: Tensor):
        return -self.softplus(-x)

class Mish(Module):
    ''' Mish activation function. '''

    def __init__(self):
        self.softplus = Softplus()
        self.tanh = Tanh()

    def forward(self, x: Tensor):
        return x * self.tanh(self.softplus(x))