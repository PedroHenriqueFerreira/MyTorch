import torch
from torch import nn

import mytorch
import mytorch.nn as nn2

layer = nn.Embedding(10, 4)

input = torch.LongTensor([[3, 3], [2, 3], [1, 1]])

y = layer(input)

print(y)

y.backward(torch.ones_like(y))

print('GRAD', layer.weight.grad)

print('#' * 20)

layer2 = nn2.Embedding(10, 4)
layer2.weight.data = layer.weight.detach().numpy()

input2 = mytorch.tensor([[3, 3], [2, 3], [1, 1]])
input2.data = input.numpy()

y2 = layer2(input2)

print(y2)

y2.backward()

print('GRAD', layer2.weight.grad)