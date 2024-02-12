from torch.nn import Softmax as Sof
import torch

from mytorch.autograd import Tensor

import numpy as np

from mytorch.nn.activations import Softmax

t1 = Tensor(np.array([[0., 1.]]), requires_grad=True)
t3 = torch.tensor([[0., 1.]], dtype=torch.float64, requires_grad=True)

f1 = Softmax()
f3 = Sof(dim=1)

y1 = f1(t1)
y3 = f3(t3)

print(id(t1))

print(y1)
print(y3)

y1.backward([[1., 2.]])
y3.backward(torch.tensor([[1., 2.]]))

print(t1.grad)
print(t3.grad)