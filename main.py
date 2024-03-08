import mytorch
from mytorch.nn import MSELoss

p1 = mytorch.tensor([0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1])
x1 = mytorch.tensor([-1000, -100, -10, -1, 0, 1, 10, 100, 1000, 1, 0], dtype=mytorch.float64, requires_grad=True)
y1 = MSELoss()(p1, x1)

print(y1)

y1.backward()

print(x1.grad)

print('-' * 50)

import torch
from torch.nn import MSELoss as TMSELoss

p2 = torch.tensor([0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1], dtype=torch.float64, requires_grad=True)
x2 = torch.tensor([-1000, -100, -10, -1, 0, 1, 10, 100, 1000, 1, 0], dtype=torch.float64, requires_grad=True)
y2 = TMSELoss()(p2, x2)

print(y2)

y2.backward(torch.ones_like(y2))

print(x2.grad)