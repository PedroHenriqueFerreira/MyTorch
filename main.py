import numpy as np

import torch

y = torch.tensor([3., 4.], requires_grad=True)
p = torch.tensor([2., 3.5], requires_grad=True)

mse = (0.5 * (y - p) ** 2).mean()

mse.backward()

print('MSE', p.grad)