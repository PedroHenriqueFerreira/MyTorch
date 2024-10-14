import mytorch
from mytorch import nn as nn2

layer2 = nn2.Conv2d(3, 3, 3, 1, 'valid', 1)

x2 = mytorch.randn(2, 3, 100, 100, requires_grad=True)
y2 = layer2(x2)

print(y2)