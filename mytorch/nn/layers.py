from abc import ABC, abstractmethod
from typing import Literal, Optional

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
    def forward(self, x: Tensor, *args) -> Tensor:
        ''' Forward pass. '''
        
        pass
    
    def __call__(self, x: Tensor, *args) -> Tensor:
        return self.forward(x, *args)
    
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

class RNNCell(Layer):
    ''' Elman RNN cell '''
    
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, nonlinearity: Literal['tanh', 'relu'] = 'tanh'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = Tanh() if nonlinearity == 'tanh' else ReLU()
        
        stdv = 1 / hidden_size ** 0.5
        
        self.weight_ih = mytorch.uniform(-stdv, stdv, (hidden_size, input_size), mytorch.float32, True)
        self.weight_hh = mytorch.uniform(-stdv, stdv, (hidden_size, hidden_size), mytorch.float32, True)
        
        if bias:
            self.bias_ih = mytorch.uniform(-stdv, stdv, (hidden_size, ), mytorch.float32, True)
            self.bias_hh = mytorch.uniform(-stdv, stdv, (hidden_size, ), mytorch.float32, True)
        else:
            self.bias_ih = None
            self.bias_hh = None
            
    def named_parameters(self):
        return [
            ('weight_ih', self.weight_ih), 
            ('weight_hh', self.weight_hh), 
            ('bias_ih', self.bias_ih), 
            ('bias_hh', self.bias_hh)
        ]

    def forward(self, x: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        if hx is None:
            hx = mytorch.zeros((x.shape[0], self.hidden_size), mytorch.float32)
        
        y = x @ self.weight_ih.T + hx @ self.weight_hh.T
        
        if self.bias_ih is not None:
            y += self.bias_ih
            
        if self.bias_hh is not None:
            y += self.bias_hh
            
        return self.nonlinearity(y)
    
class RNN(Layer):
    ''' Elman RNN '''
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_layers: int, 
        nonlinearity: Literal['tanh', 'relu'] = 'tanh', 
        bias: bool = True, 
        batch_first: bool = False, 
        dropout: float = 0,
        bidirectional: bool = False # Ignored
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = Tanh() if nonlinearity == 'tanh' else ReLU()
        self.bias = bias
        
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.cells = [RNNCell(input_size, hidden_size, bias, nonlinearity)]
        
        for _ in range(num_layers - 1):
            self.cells.append(RNNCell(hidden_size, hidden_size, bias, nonlinearity))
            
    def named_parameters(self):
        return []
    
    def forward(self, x: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        if hx is None:
            hx = mytorch.zeros((self.num_layers, x.shape[1], self.hidden_size), mytorch.float32)
        
        hidden = [hx[i] for i in range(self.num_layers)]
        
        outs: list[Tensor] = []
        
        for i in range(x.shape[0]):
            for j in range(self.num_layers):
                if j == 0:
                    hidden_l = self.cells[j](x[i], hidden[j])
                else:
                    hidden_l = self.cells[j](hidden[j - 1], hidden[j])
                    
                hidden[j] = hidden_l
            
            outs.append(hidden_l)
        
        saida =
        
        return outs, hidden