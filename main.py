import torch
from torch.nn import Linear

a = Linear(2, 2)

print(dict(a.named_parameters()))