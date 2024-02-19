from torch.nn import BCELoss

import torch

from mytorch.nn import BCELoss as BCELoss2

import mytorch

f = BCELoss(weight=torch.tensor([[2.0]]))
f2 = BCELoss2(weight=mytorch.tensor([[2.0]]))

y = torch.tensor([[1.0]], dtype=torch.float32)
y2 = mytorch.tensor([[1.0]], dtype=mytorch.float32)

p = torch.tensor([[1.0]], dtype=torch.float32, requires_grad=True)
p2 = mytorch.tensor([[1.0]], dtype=mytorch.float32, requires_grad=True)

r = f(p, y)
r2 = f2(p2, y2)

print(r)
print(r2)

r.backward()
r2.backward()

print(p.grad)
print(p2.grad)
