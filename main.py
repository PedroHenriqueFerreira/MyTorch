from mytorch.nn.losses import NLLLoss as func
from torch.nn import NLLLoss as torch_func
import torch

from mytorch.autograd import Tensor

import numpy as np

t1 = Tensor(np.array([[0.6, 0.4], [0.8, 0.2], [0.1, 0.8]]), requires_grad=True)
t2 = torch.tensor([[0.6, 0.4], [0.8, 0.2], [0.1, 0.8]], requires_grad=True)

f = func(weight=Tensor([0.3, 0.7]))
torch_f = torch_func(weight=torch.tensor([0.3, 0.7]))

y = f(t1, Tensor(np.array([1, 0, 1])))
torch_y = torch_f(t2, torch.tensor([1, 0, 1]))

print(y)
print(torch_y)

y.backward()
torch_y.backward(torch.ones_like(torch_y))

print(t1.grad)
print(t2.grad)
