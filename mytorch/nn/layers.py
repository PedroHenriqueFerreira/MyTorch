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

    def eval(self):
        ''' Sets the layer to evaluation mode. '''

        if hasattr(self, 'training'):
            self.training = False

        for item in self.__dict__.values():
            if isinstance(item, Layer):
                item.eval()

    def train(self, mode: bool = True):
        ''' Sets the layer to training mode. '''

        if hasattr(self, 'training'):
            self.training = mode

        for item in self.__dict__.values():
            if isinstance(item, Layer):
                item.train(mode)


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

    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        bias: bool = True, 
        nonlinearity: Literal['tanh', 'relu'] = 'tanh'
    ):
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

class Dropout(Layer):
    ''' Randomly zeroes some of the elements of the input tensor with probability p. '''

    def __init__(self, p: float = 0.5):
        self.p = p

        self.training = True
        self.scale = 1 / (1 - p)

    def named_parameters(self):
        return []

    def forward(self, x: Tensor):
        if self.training:
            mask = mytorch.binomial(
                1, 1 - self.p, x.shape, mytorch.float32) * self.scale
        else:
            mask = 1

        return x * mask

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
        bidirectional: bool = False  # Ignored
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias

        self.batch_first = batch_first
        self.bidirectional = bidirectional

        if dropout > 0:
            self.dropout = Dropout(dropout)
        else:
            self.dropout = None

        self.cells = [RNNCell(input_size, hidden_size, bias, nonlinearity)]
        self.cells_reverse = [RNNCell(input_size, hidden_size, bias, nonlinearity)]

        directions = 2 if bidirectional else 1

        for _ in range(num_layers - 1):
            self.cells.append(RNNCell(hidden_size * directions, hidden_size, bias, nonlinearity))
            self.cells_reverse.append(RNNCell(hidden_size * directions, hidden_size, bias, nonlinearity))

    def named_parameters(self):
        parameters: list[tuple[str, Tensor]] = []

        for i, cell in enumerate(self.cells):
            for name, parameter in cell.named_parameters():
                parameters.append((f'{name}_{i}', parameter))

        return parameters

    def forward(self, x: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        if self.batch_first:
            x = x.swapaxes(0, 1)
        
        sequence_size, batch_size, *_ = x.shape

        if hx is None:
            directions = 2 if self.bidirectional else 1
            
            hx = mytorch.zeros((directions * self.num_layers, batch_size, self.hidden_size), mytorch.float32)

        if self.bidirectional:
            hidden = [hx[i] for i in range(0, 2 * self.num_layers, 2)]
            hidden_reverse = [hx[i] for i in range(1, 2 * self.num_layers, 2)]
            
            # hidden = [hx[i] for i in range(self.num_layers)]
            # hidden_reverse = [hx[i] for i in range(self.num_layers, 2 * self.num_layers)]
            
        else:
            hidden = [hx[i] for i in range(self.num_layers)]
            hidden_reverse = None

        outputs: list[Tensor] = []
        
        if self.bidirectional:  
            outputs_reverse: list[Tensor] = []
        else:
            outputs_reverse = None

        for i in range(sequence_size):
            for layer in range(self.num_layers):
                if layer == 0:
                    hidden[layer] = self.cells[layer](x[i], hidden[layer])

                    if self.bidirectional:
                        hidden_reverse[layer] = self.cells_reverse[layer](x[-i - 1], hidden_reverse[layer])
                else:
                    prev = hidden[layer - 1].cat([hidden_reverse[layer - 1]], 1)
                    
                    hidden[layer] = self.cells[layer](prev, hidden[layer])
                    
                    if self.bidirectional:
                        # prev = hidden_reverse[layer].cat([reverse[layer - 1]], 1)
                        
                        hidden_reverse[layer] = self.cells_reverse[layer](prev, hidden_reverse[layer])

                # if layer < self.num_layers - 1 and self.dropout is not None:
                #     hidden[layer] = self.dropout(hidden[layer])
                    
                #     if self.bidirectional:
                #         hidden_reverse[layer] = self.dropout(hidden_reverse[layer])

            outputs.append(hidden[-1])
            
            if self.bidirectional:
                outputs_reverse.append(hidden_reverse[-1])

        output = mytorch.stack(outputs)
        h_n = mytorch.stack(hidden)
        
        if self.batch_first:
            output = output.swapaxes(0, 1)

        if self.bidirectional:
            output_reverse = mytorch.stack(outputs_reverse[::-1])
            h_n_reverse = mytorch.stack(hidden_reverse)
            
            if self.batch_first:
                output_reverse = output_reverse.swapaxes(0, 1)
                
            output = output.cat([output_reverse], 2)
            h_n = h_n.cat([h_n_reverse], 2)
        
        
        return output, h_n
