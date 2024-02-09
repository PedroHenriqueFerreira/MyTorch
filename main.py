# import torch
from mytorch.nn.activations import Sigmoid, SigmoidFast
from mytorch.autograd import Tensor

import numpy as np

from time import time

ini = time()

act = Sigmoid()

y = Tensor(np.random.randn(1000, 10000), requires_grad=True)
res = act(y)

print(res)

res.backward()

print(y.grad)

print("Time:", time() - ini)