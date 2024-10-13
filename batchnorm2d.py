from torch import argmin, tensor, float32, randn, ones
from torch.nn import BatchNorm2d

import mytorch
from mytorch import nn

layer = BatchNorm2d(3, affine=True)

for param in layer._parameters:
    print(param, '->', layer._parameters[param])

data = randn(2, 3, 2, 3)

output = layer(data)
output = layer(output)
layer.eval()
output = layer(output)

# print('D', data)

print('O', output)

print('M', layer.running_mean)
print('V', layer.running_var)

# print('W', layer._parameters['weight'])
# print('B', layer._parameters['bias'])

print('-' * 50)

layer2 = nn.BatchNorm2d(3, affine=True)

if layer2.affine:
    layer2.weight.data = layer._parameters['weight'].detach().numpy()
    layer2.bias.data = layer._parameters['bias'].detach().numpy()

# layer2.running_mean.data = layer.running_mean.detach().numpy()
# layer2.running_var.data = layer.running_var.detach().numpy()

data2 = mytorch.randn(2, 3, 2, 3)
data2.data = data.detach().numpy()

output2 = layer2(data2)
output2 = layer2(output2)

layer2.eval()
output2 = layer2(output2)

# print('D2', data2)

print('O2', output2)

print('M2', layer2.running_mean)
print('V2', layer2.running_var)

# print('W2', layer2.weight)
# print('B2', layer2.bias)