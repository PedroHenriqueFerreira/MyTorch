from mytorch.nn.activations import SELU as func
from torch.nn import SELU as torch_func
import torch

from mytorch.autograd import Tensor

import numpy as np

t1 = Tensor(np.array([[-1., -0.5, 0, 0.5, 1.]]), requires_grad=True)
t2 = torch.tensor([[-1., -0.5, 0, 0.5, 1.]], requires_grad=True)

f = func()
torch_f = torch_func()

y = f(t1)
torch_y = torch_f(t2)

print(y)
print(torch_y)

y.backward()
torch_y.backward(torch.ones_like(torch_y))

print(t1.grad)
print(t2.grad)
