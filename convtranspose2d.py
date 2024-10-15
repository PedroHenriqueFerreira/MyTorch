import torch
from torch import nn

import mytorch
import mytorch.nn as nn2

layer = nn.ConvTranspose2d(3, 1, 2, 1, 0, 0)
x = torch.randn(1, 3, 3, 3, requires_grad=True)

y = layer(x)

print(y)

print('#' * 20)

layer2 = nn2.ConvTranspose2d(3, 1, 2, 1, 0, 0)
layer2.weight.data = layer._parameters['weight'].detach().numpy()
layer2.bias.data = layer._parameters['bias'].detach().numpy()

x2 = mytorch.randn(1, 3, 3, 3, requires_grad=True)
x2.data = x.detach().numpy()

y2 = layer2(x2)

print(y2)