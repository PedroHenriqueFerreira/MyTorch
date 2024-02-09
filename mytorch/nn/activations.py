from abc import abstractmethod

from mytorch.autograd import Tensor

import numpy as np

class Activation(Tensor):
    ''' Base class for all loss functions. '''
    
    def __init__(self):
        ''' Initializes the activation function. '''
        
        pass
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        ''' Forward pass. '''
        
        pass
    
    def __call__(self, x: Tensor) -> Tensor:
        ''' Forward pass. '''
        
        return self.forward(x)
    
class Sigmoid(Activation):
    ''' Sigmoid activation function. '''
    
    def forward(self, x: Tensor) -> Tensor:
        ''' Forward pass. '''
        
        return 1 / (1 + (-x).exp())

class SigmoidFast:
    ''' Fast sigmoid activation function. '''
    
    def forward(self, x: Tensor) -> Tensor:
        ''' Forward pass. '''
        
        f_x = 1 / (1 + np.exp(-x.data))
        
        def grad_fn(grad: np.ndarray):
            x.backward(grad * f_x * (1 - f_x))
        
        return Tensor(f_x, requires_grad=True, grad_fn=grad_fn)
    
    def __call__(self, x: Tensor) -> Tensor:
        ''' Forward pass. '''
        
        return self.forward(x)