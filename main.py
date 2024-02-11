from torch.nn import Softmax as Sof
import torch

from mytorch.autograd import Tensor

import numpy as np

from mytorch.nn.activations import Softmax, Softmax2

t1 = Tensor(np.array([[0, 1], [2, 3], [4, 5]]), requires_grad=True)
t2 = Tensor(np.array([[0, 1], [2, 3], [4, 5]]), requires_grad=True)

t3 = torch.tensor([[0., 1.], [2., 3.], [4., 5.]],  requires_grad=True)

f1 = Softmax()
f2 = Softmax2()
f3 = Sof(dim=1)

y1 = f1(t1)
y2 = f2(t2)
y3 = f3(t3)

print(y1)
print(y2)
print(y3)

y1.backward([[1, 2], [3, 4], [5, 6]])
y2.backward([[1, 2], [3, 4], [5, 6]])
y3.backward(torch.tensor([[1, 2], [3, 4], [5, 6]]))

print(t1.grad)
print(t2.grad)
print(t3.grad)