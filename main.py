import mytorch
from mytorch.nn import Softmax

x1 = mytorch.tensor([[-10000, -1000, 1, 0, -200, -100, -50, -20, -10], [-5, -1, 1, 0, 0, 1, 20, 1000, 1000]], dtype=mytorch.float64, requires_grad=True)
y1 = Softmax()(x1)

print(y1)

y1.backward()

print(x1.grad)

print('-' * 50)

import torch
from torch.nn import Softmax as TSoftmax

x2 = torch.tensor([[-10000, -1000, 1, 0, -200, -100, -50, -20, -10], [-5, -1, 1, 0, 0, 1, 20, 1000, 1000]], dtype=torch.float64, requires_grad=True)
y2 = TSoftmax()(x2)

print(y2)

y2.backward(torch.ones_like(y2))

print(x2.grad)