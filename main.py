import numpy as np

import torch

y = torch.tensor(3, requires_grad=True)

mse = (0.5 * (y - p) ** 2).mean()

mse.backward()

print('MSE', p.grad)