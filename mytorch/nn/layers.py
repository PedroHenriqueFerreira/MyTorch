from abc import ABC, abstractmethod

import mytorch
from mytorch import Tensor

class Layer(ABC):
    ''' Abstract class for layers. '''
    
    @abstractmethod
    def named_parameters(self) -> list[tuple[str, Tensor]]:
        ''' Returns the parameters of the layer. '''
        
        pass
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        ''' Forward pass. '''
        
        pass
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
class Linear(Layer):
    ''' Linear layer. '''

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features

        stdv = 1. / in_features ** 0.5
        
        self.weight = mytorch.uniform(-stdv, stdv, (out_features, in_features), mytorch.float32, True)
        
        if bias:
            self.bias = mytorch.uniform(-stdv, stdv, (1, out_features), mytorch.float32, True)
        else:
            self.bias = None
    
    def named_parameters(self):
        return [('weight', self.weight), ('bias', self.bias)]
    
    def forward(self, x: Tensor) -> Tensor:
        y = x @ self.weight.T
        
        if self.bias is not None:
            y += self.bias
            
        return y
    
