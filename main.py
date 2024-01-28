import numpy as np

import torch

m = torch.tensor([3., 4.], requires_grad=True)
v = torch.tensor([1., 2.], requires_grad=True)

r = v @ m

print('SAIDA', r)

r.backward(torch.tensor(1.))

print('M GRAD', m.grad)
print('V GRAD', v.grad)