import mytorch
from mytorch import nn as nn2

from torch import nn
import torch

INPUT_SIZE = 2 # OK
HIDDEN_SIZE = 2 # OK 
LAYERS = 2 #
BATCHES = 2 # OK
SEQUENCE_SIZE = 2 # OK

# Entrada, Escondidos, Camadas
lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, LAYERS, True, False)

# print([a for a in rnn._parameters])
for a in lstm._parameters:
    print(a, '->', lstm._parameters[a].shape)

# Lotes, Momentos, Entradas
input = torch.randn(SEQUENCE_SIZE, BATCHES, INPUT_SIZE)

# Camadas, Lotes, Escondidos
h0 = torch.randn(LAYERS, BATCHES, HIDDEN_SIZE, requires_grad=True)
c0 = torch.randn(LAYERS, BATCHES, HIDDEN_SIZE, requires_grad=True)

output, (hn, cn) = lstm(input, (h0, c0))

print('O ->', output)
print('H ->', hn)
print('C ->', cn)

output.backward(torch.ones_like(output))

print('G ->', h0.grad)

print('-' * 50)

lstm2 = nn2.LSTM(INPUT_SIZE, HIDDEN_SIZE, LAYERS, True, False)

for layer in range(LAYERS):
    lstm2.layers[layer].weight_ih.data = lstm._parameters[f'weight_ih_l{layer}'].detach().numpy()
    lstm2.layers[layer].weight_hh.data = lstm._parameters[f'weight_hh_l{layer}'].detach().numpy()
    
    if lstm.bias:
        lstm2.layers[layer].bias_ih.data = lstm._parameters[f'bias_ih_l{layer}'].detach().numpy()
        lstm2.layers[layer].bias_hh.data = lstm._parameters[f'bias_hh_l{layer}'].detach().numpy()

input2 = mytorch.randn(SEQUENCE_SIZE, BATCHES, INPUT_SIZE)
input2.data = input.detach().numpy()

h02 = mytorch.randn(LAYERS, BATCHES, HIDDEN_SIZE, requires_grad=True)
h02.data = h0.detach().numpy()

c02 = mytorch.randn(LAYERS, BATCHES, HIDDEN_SIZE, requires_grad=True)
c02.data = c0.detach().numpy()

output2, (hn2, cn02) = lstm2(input2, (h02, c02))

print('O ->', output2)
print('H ->', hn2)
print('C ->', cn02)

output2.backward()

print('G ->', h02.grad)