from mytorch.autograd import Tensor

import numpy as np

from mytorch.nn.activations import Softmax, Softmax2

t1 = Tensor(np.array([[0, 1], [2, 3], [4, 5]]), requires_grad=True)
t2 = Tensor(np.array([[0, 1], [2, 3], [4, 5]]), requires_grad=True)

f1 = Softmax()
f2 = Softmax2()

y1 = f1(t1)
y2 = f2(t2)

print(y1)
print(y2)

y1.backward([[1, 2], [3, 4], [5, 6]])
y2.backward([[1, 2], [3, 4], [5, 6]])

print(t1.grad)
print(t2.grad)