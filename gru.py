import mytorch
from mytorch import nn as nn2

from torch import nn
import torch

INPUT_SIZE = 2 # OK
HIDDEN_SIZE = 1 # OK 
LAYERS = 5 #
BATCHES = 2 # OK
SEQUENCE_SIZE = 3 # OK

# Entrada, Escondidos, Camadas
gru = nn.GRU(INPUT_SIZE, HIDDEN_SIZE, LAYERS, True, False)

# print([a for a in gru._parameters])
for a in gru._parameters:
    print(a, '->', gru._parameters[a].shape)

# Lotes, Momentos, Entradas
input = torch.randn(SEQUENCE_SIZE, BATCHES, INPUT_SIZE)

# Camadas, Lotes, Escondidos
h0 = torch.randn(LAYERS, BATCHES, HIDDEN_SIZE, requires_grad=True)

output, hn = gru(input, h0)

print(output)

output.backward(torch.ones_like(output))

print(h0.grad)

print('-' * 50)

gru2 = nn2.GRU(INPUT_SIZE, HIDDEN_SIZE, LAYERS, True, False)

for layer in range(LAYERS):
    gru2.layers[layer].weight_ih.data = gru._parameters[f'weight_ih_l{layer}'].detach().numpy()
    gru2.layers[layer].weight_hh.data = gru._parameters[f'weight_hh_l{layer}'].detach().numpy()
    
    if gru.bias:
        gru2.layers[layer].bias_ih.data = gru._parameters[f'bias_ih_l{layer}'].detach().numpy()
        gru2.layers[layer].bias_hh.data = gru._parameters[f'bias_hh_l{layer}'].detach().numpy()

input2 = mytorch.randn(SEQUENCE_SIZE, BATCHES, INPUT_SIZE)
input2.data = input.detach().numpy()

h02 = mytorch.randn(LAYERS, BATCHES, HIDDEN_SIZE, requires_grad=True)
h02.data = h0.detach().numpy()

output2, hn2 = gru2(input2, h02)

print(output2)

output2.backward()

print(h02.grad)