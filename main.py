import mytorch
from mytorch.nn import GELU

x1 = mytorch.tensor([-10000, -1000, -200, -100, -50, -20, -10, -5, -1, 0, 1, 20, 1000, 1000, 1000], dtype=mytorch.float64, requires_grad=True)
y1 = GELU()(x1)

print(y1)

y1.backward()

print(x1.grad)

print('-' * 50)

import torch
from torch.nn import GELU as TGELU

x2 = torch.tensor([-10000, -1000, -200, -100, -50, -20, -10, -5, -1, 0, 1, 20, 1000, 1000, 1000], dtype=torch.float64, requires_grad=True)
y2 = TGELU()(x2)

print(y2)

y2.backward(torch.ones_like(y2))

print(x2.grad)