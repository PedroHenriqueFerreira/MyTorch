from mytorch.nn.activations import ReLU as func

from mytorch.autograd import Tensor

import numpy as np

t1 = Tensor(np.array([[-1., -0.5, 0, 0.5, 1.]]), requires_grad=True)

f = func()

y = f(t1)

print(y)

y.backward()

print(t1.grad)