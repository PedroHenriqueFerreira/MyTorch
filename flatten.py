import torch
from torch import nn

import mytorch
import mytorch.nn as nn2

layer = nn.Flatten(start_dim=2, end_dim=3)

x = torch.randn(2, 3, 2, 2, requires_grad=True)

y = layer(x)

print(y)

print('#' * 20)

layer2 = nn2.Flatten(start_dim=2, end_dim=3)

x2 = mytorch.randn(1, 3, 4, 4, requires_grad=True)
x2.data = x.detach().numpy()

y2 = layer2(x2)

print(y2)

