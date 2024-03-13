import torch
from torch.nn import Linear

torch.no_grad

a = Linear(2, 2)

a.eval()

print(dict(a.named_parameters()))
print(a.weight)
print(a.bias)

y = a(torch.tensor([1., 2.]))

print('-----------------')
print(y)