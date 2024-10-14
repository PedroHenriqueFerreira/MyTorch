import torch
from torch import nn

import mytorch
from mytorch import nn as nn2

layer = nn.Conv2d(3, 2, 3, 1, 'valid', 2)

x = torch.randn(1, 3, 5, 5, requires_grad=True)

y = layer(x)
print(y)

y.backward(torch.ones_like(y))

print(x.grad)

print('#' * 20)

layer2 = nn2.Conv2d(3, 2, 3, 1, 'valid', 2)

layer2.weight.data = layer._parameters['weight'].detach().numpy()
layer2.bias.data = layer._parameters['bias'].detach().numpy()

x2 = mytorch.randn(1, 3, 5, 5, requires_grad=True)
x2.data = x.detach().numpy()

y2 = layer2(x2)

print(y2)

y2.backward()

print(x2.grad)