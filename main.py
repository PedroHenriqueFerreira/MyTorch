import numpy as np

import torch
from torch.nn import L1Loss
from torch.optim import adadelta
# from losses import L1Loss as L1Loss2

loss = L1Loss()
loss2 = L1Loss2()

a = torch.tensor([1, 2., 3.])
b = torch.tensor([1., 3., 3.], requires_grad=True).backward()

output = loss(a, b)

print(output)

output.backward()

print(a.grad, b.grad)

print(loss2(np.array([1, 2, 3]), np.array([1, 3, 3])))