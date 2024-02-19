from torch.nn import BCEWithLogitsLoss

import torch

from mytorch.nn import BCEWithLogitsLoss as BCEWithLogitsLoss2

import mytorch

f = BCEWithLogitsLoss(weight=torch.tensor([[2.0]]))
f2 = BCEWithLogitsLoss2(weight=mytorch.tensor([[2.0]]))

y = torch.tensor([[1.0]], dtype=torch.float32)
y2 = mytorch.tensor([[1.0]], dtype=mytorch.float32)

p = torch.tensor([[-2000]], dtype=torch.float32, requires_grad=True)
p2 = mytorch.tensor([[-2000]], dtype=mytorch.float32, requires_grad=True)

r = f(p, y)
r2 = f2(p2, y2)

print(r)
print(r2)

r.backward()
r2.backward()

print(p.grad)
print(p2.grad)
