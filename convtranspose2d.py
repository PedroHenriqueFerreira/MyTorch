import torch
from torch import nn

import mytorch
import mytorch.nn as nn2

layer = nn.ConvTranspose2d(3, 2, 3, 1, 3, 2, dilation=3, bias=True)

x = torch.randn(1, 3, 4, 4, requires_grad=True)

y = layer(x)

print(y)

y.backward(torch.ones_like(y))

print('GRAD', x.grad)

print('#' * 20)

layer2 = nn2.ConvTranspose2d(3, 2, 3, 1, 3, 2, dilation=3, bias=True)
layer2.weight.data = layer._parameters['weight'].detach().numpy()
layer2.bias.data = layer._parameters['bias'].detach().numpy()

x2 = mytorch.randn(1, 3, 4, 4, requires_grad=True)
x2.data = x.detach().numpy()

y2 = layer2(x2)

print(y2)

y2.backward()

print('GRAD2', x2.grad)