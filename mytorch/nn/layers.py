from typing import Literal, Optional

import mytorch
from mytorch import Tensor, DeviceLikeType
from mytorch.nn import ReLU, Tanh
from mytorch.nn.modules import Module

class Linear(Module):
    ''' Linear layer. '''

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: DeviceLikeType = 'cpu'):
        self.in_features = in_features
        self.out_features = out_features

        stdv = 1. / in_features ** 0.5

        self.weight = mytorch.uniform(-stdv, stdv, (out_features, in_features), mytorch.float32, True)

        if bias:
            self.bias = mytorch.uniform(-stdv, stdv, (1, out_features), mytorch.float32, True)
        else:
            self.bias = None
            
        self.to(device)

    def forward(self, x: Tensor) -> Tensor:
        if self.device != x.device:
            raise ValueError('Tensors must be on the same device')
        
        y = x @ self.weight.T

        if self.bias is not None:
            y += self.bias

        return y

class RNNCell(Module):
    ''' Elman RNN cell '''

    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        bias: bool = True, 
        nonlinearity: Literal['tanh', 'relu'] = 'tanh',
        device: DeviceLikeType = 'cpu'
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
            
        self.to(device)

    def forward(self, x: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        if self.device != x.device:
            raise ValueError('Tensors must be on the same device')
        
        if hx is not None and self.device != hx.device:
            raise ValueError('Tensors must be on the same device')
        
        if hx is None:
            hx = mytorch.zeros((x.shape[0], self.hidden_size), mytorch.float32)
        
        y = x @ self.weight_ih.T + hx @ self.weight_hh.T

        if self.bias_ih is not None:
            y += self.bias_ih

        if self.bias_hh is not None:
            y += self.bias_hh

        return self.nonlinearity(y)

class Dropout(Module):
    ''' Randomly zeroes some of the elements of the input tensor with probability p. '''

    def __init__(self, p: float = 0.5):
        self.p = p
        
        self.scale = 1 / (1 - p)
        
        self.training = True

    def forward(self, x: Tensor):
        if self.training:
            mask = mytorch.binomial(1, 1 - self.p, x.shape, x.dtype, False, x.device) * self.scale
        else:
            mask = 1

        return x * mask
    
    def eval(self):
        self.training = False
        
    def train(self, mode: bool = True):
        self.training = mode

class RNN(Module):
    ''' Elman RNN '''

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: Literal['tanh', 'relu'] = 'tanh',
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        device: DeviceLikeType = 'cpu'
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first

        self.layers = [RNNCell(input_size, hidden_size, bias, nonlinearity, device)]
        
        for _ in range(num_layers - 1):
            self.layers.append(RNNCell(hidden_size, hidden_size, bias, nonlinearity, device))

        self.dropout = Dropout(dropout)

        self.to(device)

    def forward(self, x: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        if self.device != x.device:
            raise ValueError('Tensors must be on the same device')
        
        if hx is not None and self.device != hx.device:
            raise ValueError('Tensors must be on the same device')
        
        if self.batch_first:
            x = x.swapaxes(0, 1)
        
        sequence_size, batch_size = x.shape[:2]

        if hx is None:
            hx = mytorch.zeros((self.num_layers, batch_size, self.hidden_size), mytorch.float32)

        hidden = [hx[i] for i in range(self.num_layers)]

        outputs: list[Tensor] = []
        
        for sequence in range(sequence_size):
            for layer in range(self.num_layers):
                input = x[sequence] if layer == 0 else hidden[layer - 1]
                
                hidden[layer] = self.layers[layer](input, hidden[layer])

                if layer < self.num_layers - 1:
                    hidden[layer] = self.dropout(hidden[layer])
                
            outputs.append(hidden[-1])

        output = mytorch.stack(outputs)
        hn = mytorch.stack(hidden)
        
        if self.batch_first:
            output = output.swapaxes(0, 1)

        return output, hn

class LSTMCell(Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, device: DeviceLikeType = 'cpu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        stdv = 1 / hidden_size ** 0.5

        self.weight_ih = mytorch.uniform(-stdv, stdv, (4 * hidden_size, input_size), mytorch.float32, True)
        self.weight_hh = mytorch.uniform(-stdv, stdv, (4 * hidden_size, hidden_size), mytorch.float32, True)

        if bias:
            self.bias_ih = mytorch.uniform(-stdv, stdv, (4 * hidden_size, ), mytorch.float32, True)
            self.bias_hh = mytorch.uniform(-stdv, stdv, (4 * hidden_size, ), mytorch.float32, True)
        else:
            self.bias_ih = None
            self.bias_hh = None
        
        self.to(device)
        
    def forward(self, x: Tensor, hx: Optional[tuple[Tensor, Tensor]] = None) -> tuple[Tensor, Tensor]:
        if self.device != x.device:
            raise ValueError('Tensors must be on the same device')
        
        if hx is not None and (self.device != hx[0].device or self.device != hx[1].device):
            raise ValueError('Tensors must be on the same device')
        
        if hx is None:
            hx = (
                mytorch.zeros((x.shape[0], self.hidden_size), mytorch.float32),
                mytorch.zeros((x.shape[0], self.hidden_size), mytorch.float32)
            )
        
        y = x @ self.weight_ih.T + hx[0] @ self.weight_hh.T
        
        if self.bias_ih and self.bias_hh:
            y += self.bias_ih + self.bias_hh
        
        i = y[:, 0 * self.hidden_size : 1 * self.hidden_size].sigmoid()
        f = y[:, 1 * self.hidden_size : 2 * self.hidden_size].sigmoid()
        g = y[:, 2 * self.hidden_size : 3 * self.hidden_size].tanh()
        o = y[:, 3 * self.hidden_size : 4 * self.hidden_size].sigmoid()
        
        c = f * hx[1] + i * g
        h = o * c.tanh()
        
        return h, c
    
class LSTM(Module):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_layers = 1, 
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        device: DeviceLikeType = 'cpu'
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first

        self.layers = [LSTMCell(input_size, hidden_size, bias, device)]
        
        for _ in range(num_layers - 1):
            self.layers.append(LSTMCell(hidden_size, hidden_size, bias, device))

        self.dropout = Dropout(dropout)
        
        self.to(device)
        
    def forward(self, x: Tensor, hx: Optional[tuple[Tensor, Tensor]] = None) -> tuple[Tensor, Tensor]:
        if self.device != x.device:
            raise ValueError('Tensors must be on the same device')
        
        if hx is not None and (self.device != hx[0].device or self.device != hx[1].device):
            raise ValueError('Tensors must be on the same device')
        
        if self.batch_first:
            x = x.swapaxes(0, 1)
        
        sequence_size, batch_size = x.shape[:2]
        
        if hx is None:
            hx = (
                mytorch.zeros((self.num_layers, batch_size, self.hidden_size), mytorch.float32),
                mytorch.zeros((self.num_layers, batch_size, self.hidden_size), mytorch.float32)
            )
            
        hidden = [(hx[0][i], hx[1][i]) for i in range(self.num_layers)]
        
        outputs: list[tuple[Tensor, Tensor]] = []
        
        for sequence in range(sequence_size):
            for layer in range(self.num_layers):
                input = x[sequence] if layer == 0 else hidden[layer - 1][0]
                
                hidden[layer] = self.layers[layer](input, hidden[layer])

                if layer < self.num_layers - 1:
                    hidden[layer] = (self.dropout(hidden[layer][0]), self.dropout(hidden[layer][1]))
                
            outputs.append(hidden[-1])
        
        output = mytorch.stack([output[0] for output in outputs])
        hn = mytorch.stack([item[0] for item in hidden])
        cn = mytorch.stack([item[1] for item in hidden])
        
        if self.batch_first:
            output = output.swapaxes(0, 1)
        
        return output, (hn, cn)
    
class BatchNorm2d(Module):
    def __init__(
        self,
        num_features: int, 
        eps: float = 1e-05, 
        momentum: float = 0.1, 
        affine: bool = True, 
        device: DeviceLikeType = 'cpu'
    ):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        
        if affine:
            self.weight = mytorch.ones((num_features, ), mytorch.float32, True)
            self.bias = mytorch.zeros((num_features, ), mytorch.float32, True)
        else:
            self.weight = None
            self.bias = None
        
        self.running_mean = mytorch.zeros((num_features, ), mytorch.float32)
        self.running_var = mytorch.ones((num_features, ), mytorch.float32)
        
        self.training = True
        
        self.to(device)
    
    def forward(self, x: Tensor):
        if self.device != x.device:
            raise ValueError('Tensors must be on the same device')
        
        if self.training:
            mean = x.mean(dim=(0, 2, 3))     
            var = x.var(dim=(0, 2, 3), correction=0) 
            
            var_sample = var * (x.size / (x.size - self.num_features))

            self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
            self.running_var = self.momentum * var_sample + (1 - self.momentum) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var
            
        output = (x - mean[None, :, None, None]) / (var[None, :, None, None] + self.eps).sqrt()
        
        if self.affine:
            output = output * self.weight[None, :, None, None] + self.bias[None, :, None, None]
            
        return output
    
class Conv2d(Module):
    def __init__(
        self,    
        in_channels: int, 
        out_channels: int, 
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1, 
        padding: int | tuple[int, int] = 0, 
        dilation: int | tuple[int, int] = 1, 
        bias: bool = True, 
        device: DeviceLikeType = 'cpu'
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
    
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        
        stdv = 1 / (in_channels * self.kernel_size[0] * self.kernel_size[1]) ** 0.5
        
        self.weight = mytorch.uniform(
            -stdv, 
            stdv, 
            (out_channels, in_channels, *self.kernel_size), 
            mytorch.float32, 
            True
        )
        
        if bias:
            self.bias = mytorch.uniform(
                -stdv, 
                stdv, 
                (out_channels, ), 
                mytorch.float32, 
                True
            )
        
        
        self.to(device)
    
    def image_to_col(self, x: Tensor) -> Tensor:
        ...
    
    def forward(self, x: Tensor) -> Tensor:
        if self.device != x.device:
            raise ValueError('Tensors must be on the same device')
        
        