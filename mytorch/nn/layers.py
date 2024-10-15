from typing import Union, Literal, Optional

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
        kernel_size: Union[int, tuple[int, int]],
        stride: Union[int, tuple[int, int]] = 1, 
        padding: Union[int, tuple[int, int], Literal['valid', 'same']] = 0,
        dilation: Union[int, tuple[int, int]] = 1, 
        bias: bool = True, 
        device: DeviceLikeType = 'cpu'
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
    
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        
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
        else:
            self.bias = None
        
        self.to(device)
    
    def forward(self, x: Tensor) -> Tensor:
        if self.device != x.device:
            raise ValueError('Tensors must be on the same device')
        
        batch_size, in_channels, x_h, x_w = x.shape
        
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        d_h, d_w = self.dilation
        
        # Calculating dilated kernel size
        
        di_h = d_h * (k_h - 1) + 1
        di_w = d_w * (k_w - 1) + 1
        
        # Calculating padding size
        
        if self.padding == 'valid':
            pad_h, pad_w = (0, 0), (0, 0)
        else:
            if self.padding == 'same':
                pad_h_sum = d_h * (k_h - 1) - s_h + 1
                pad_w_sum = d_w * (k_w - 1) - s_w + 1
                
                pad_h_0 = pad_h_sum // 2
                pad_h_1 = pad_h_sum - pad_h_0
                 
                pad_w_0 = pad_w_sum // 2
                pad_w_1 = pad_w_sum - pad_w_0
            
                pad_h = abs(pad_h_0), abs(pad_h_1)
                pad_w = abs(pad_w_0), abs(pad_w_1)
            
            else:
                pad_h = self.padding[0], self.padding[0]
                pad_w = self.padding[1], self.padding[1]
            
        # Calculating output size
        
        out_h = (x_h + pad_h[0] + pad_h[1] - d_h * (k_h - 1) - 1) // s_h + 1       
        out_w = (x_w + pad_w[0] + pad_w[1] - d_w * (k_w - 1) - 1) // s_w + 1
        
        out_shape = (batch_size, self.out_channels, out_h, out_w)
        
        # Applying padding

        x = x.pad(((0, 0), (0, 0), pad_h, pad_w))
        
        # Transforming image to column
        
        x_col = mytorch.zeros((batch_size, out_h * out_w, in_channels, k_h, k_w), x.dtype, True, self.device)
        
        for i in range(out_h):
            for j in range(out_w):
                pos = i * out_w + j
                
                i_start, j_start = i * s_h, j * s_w
                i_end, j_end = i_start + di_h, j_start + di_w
                
                x_col[:, pos, :, :, :] = x[:, :, i_start : i_end : d_h, j_start : j_end : d_w]
                
        x_col = x_col.reshape((batch_size, out_h * out_w, -1))   
        weight_col = self.weight.reshape((self.out_channels, -1)).T
        
        out_col = x_col @ weight_col
        
        if self.bias is not None:
            out_col += self.bias[None, None, :]
        
        # Reshaping output
        
        out = out_col.transpose((0, 2, 1)).reshape(out_shape)
        
        return out

class ConvTranspose2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, int]],
        stride: Union[int, tuple[int, int]] = 1,
        padding: Union[int, tuple[int, int]] = 0,
        output_padding: Union[int, tuple[int, int]] = 0,
        bias: bool = True,
        dilation: Union[int, tuple[int, int]] = 1,
        device: DeviceLikeType = 'cpu'
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.output_padding = (output_padding, output_padding) if isinstance(output_padding, int) else output_padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        
        stdv = 1 / (out_channels * self.kernel_size[0] * self.kernel_size[1]) ** 0.5
        
        self.weight = mytorch.uniform(
            -stdv, 
            stdv, 
            (in_channels, out_channels, *self.kernel_size), 
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
        else:
            self.bias = None
        
        self.to(device)
        
    def forward(self, x: Tensor):
        if self.device != x.device:
            raise ValueError('Tensors must be on the same device')
        
        batch_size, in_channels, x_h, x_w = x.shape
        
        opad_h, opad_w = self.output_padding
        
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        d_h, d_w = self.dilation
        
        # Calculating dilated kernel size
        
        di_h = d_h * (k_h - 1) + 1 
        di_w = d_w * (k_w - 1) + 1
        
        # Calculating padding size
        
        pad_h = self.padding[0], self.padding[0]
        pad_w = self.padding[1], self.padding[1]

        out_h = (x_h - 1) * s_h - (pad_h[0] + pad_h[1]) + di_h + opad_h
        out_w = (x_w - 1) * s_w - (pad_w[0] + pad_w[1]) + di_w + opad_w
        
        out_shape = (batch_size, self.out_channels, out_h, out_w)
        
        x = x.pad(((0, 0), (0, 0), pad_h, pad_w))
        
        x_col = x.reshape((batch_size, in_channels, -1)).transpose((0, 2, 1))
        weight_col = self.weight.reshape((in_channels, -1)).T
        
        print('XX', x_col.shape)
        print('WW', weight_col.T.shape)
        
        out_col = x_col @ weight_col.T

        print('A', out_col.shape)
        
        out_col = out_col.reshape((out_col.shape[0], out_col.shape[1], self.out_channels, k_h, k_w))
        
        out = mytorch.zeros(out_shape, x.dtype, True, self.device)
        
        for pos in range(out_h * out_w):
            i = pos // out_h
            j = pos % out_h
            
            i_start, j_start = i * s_h, j * s_w
            i_end, j_end = i_start + di_h, j_start + di_w
            
            out[:, :, i_start : i_end : d_h, j_start : j_end : d_w] += out_col[:, pos, :, :, :]
            
        return out