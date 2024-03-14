from abc import ABC, abstractmethod
from typing import Literal

import mytorch
from mytorch import Tensor
from mytorch.nn import ReLU, Tanh

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
    
class RNN(Layer):
    ''' Recurrent Neural Network layer. '''
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        nonlinearity: Literal['tanh', 'relu'] = 'tanh',
        bias: bool = True,
        cycled_states: bool = False,
        return_sequences: Literal['all', 'last', 'both'] = 'both'
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = (Tanh() if nonlinearity == 'tanh' else ReLU())
        self.cycled_states = cycled_states
        self.return_sequences = return_sequences
        
        stdv = 1 / hidden_size ** 0.5
        
        self.weight = mytorch.uniform(-stdv, stdv, (hidden_size, input_size), mytorch.float32, True)
        self.