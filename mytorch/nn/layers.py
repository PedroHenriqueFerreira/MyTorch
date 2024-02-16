from abc import ABC, abstractmethod

import mytorch as mt

from math import sqrt

class Layer(ABC):
    ''' Abstract class for layers. '''
    
    @abstractmethod
    def forward(self, x: mt.Tensor) -> mt.Tensor:
        ''' Forward pass. '''
        
        pass
    
    def __call__(self, x: mt.Tensor) -> mt.Tensor:
        return self.forward(x)
    
class Linear(Layer):
    ''' Linear layer. '''
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        
        stdv = 1. / sqrt(in_features)
        
        self.weight = mt.uniform(
            -stdv, 
            stdv, 
            (out_features, in_features), 
            dtype=mt.float32, 
            requires_grad=True
        )
        
        if bias:
            self.bias = mt.zeros((1, out_features), dtype=mt.float32, requires_grad=True)
        else:
            self.bias = None
    
    def forward(self, x: mt.Tensor) -> mt.Tensor:
        y = x @ self.weight.T
        
        if self.bias is not None:
            y += self.bias
            
        return y