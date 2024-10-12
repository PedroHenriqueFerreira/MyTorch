from torch import argmin, tensor, float32, randn
from torch.nn import BatchNorm2d

import mytorch
from mytorch import nn

layer = BatchNorm2d(2)

for param in layer._parameters:
    print(param, '->', layer._parameters[param])

data = randn(2, 2, 1, 1)

output = layer(data)

print('D', data)

print('O', output)


print('-' * 50)

layer2 = nn.BatchNorm2d(2)
layer2.weight.data = layer._parameters['weight'].detach().numpy()
layer2.bias.data = layer._parameters['bias'].detach().numpy()

# layer2.running_mean.data = layer.running_mean.detach().numpy()
# layer2.running_var.data = layer.running_var.detach().numpy()

data2 = mytorch.randn(2, 2, 1, 1)
data2.data = data.detach().numpy()

output2 = layer2(data2)

print('O2', output2)

print('M', layer2.running_mean)
print('V', layer2.running_var)