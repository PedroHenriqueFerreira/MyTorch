import mytorch
from mytorch.nn import ReLU

x1 = mytorch.tensor([-1000, 0, 1, 1000, 1000, 1000], dtype=mytorch.float64, requires_grad=True)
y1 = ReLU()(x1)

print(y1)

y1.backward()

print(x1.grad)

print('-' * 50)

import torch
from torch.nn import ReLU as TReLU

x2 = torch.tensor([-1000, 0, 1, 1000, 1000, 1000], dtype=torch.float64, requires_grad=True)
y2 = TReLU()(x2)

print(y2)

y2.backward(torch.ones_like(y2))

print(x2.grad)