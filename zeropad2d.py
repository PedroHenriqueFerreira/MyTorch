import torch
from torch import nn

import mytorch
import mytorch.nn as nn2

l = nn.ZeroPad2d((1, 2, 3, 4))

inp = torch.randn(1, 1, 3, 3, requires_grad=True)

out = l(inp)

print(out)

out.backward(torch.ones_like(out))

print(inp.grad)

print("#" * 20)

l2 = nn2.ZeroPad2d((1, 2, 3, 4))

inp2 = mytorch.randn(1, 1, 3, 3, requires_grad=True)
inp2.data = inp.detach().numpy()

out2 = l2(inp2)

print(out2)

out2.backward()

print(inp2.grad)