import mytorch
from mytorch import nn as nn2

from torch import nn
import torch

INPUT_SIZE = 1 # OK
HIDDEN_SIZE = 1 # OK 
LAYERS = 5 #
BATCHES = 2 # OK
SEQUENCE_SIZE = 2 # OK

# Entrada, Escondidos, Camadas
rnn = nn.RNN(INPUT_SIZE, HIDDEN_SIZE, LAYERS, 'tanh', True, False)

# print([a for a in rnn._parameters])
for a in rnn._parameters:
    print(a, '->', rnn._parameters[a].shape)

# Lotes, Momentos, Entradas
input = torch.randn(SEQUENCE_SIZE, BATCHES, INPUT_SIZE)

# Camadas, Lotes, Escondidos
h0 = torch.randn(LAYERS, BATCHES, HIDDEN_SIZE, requires_grad=True)

output, hn = rnn(input, h0)

print(output)

output.backward(torch.ones_like(output))

print(h0.grad)

print('-' * 50)

rnn2 = nn2.RNN(INPUT_SIZE, HIDDEN_SIZE, LAYERS, 'tanh', True, False)

for layer in range(LAYERS):
    rnn2.layers[layer].weight_ih.data = rnn._parameters[f'weight_ih_l{layer}'].detach().numpy()
    rnn2.layers[layer].weight_hh.data = rnn._parameters[f'weight_hh_l{layer}'].detach().numpy()
    
    if rnn.bias:
        rnn2.layers[layer].bias_ih.data = rnn._parameters[f'bias_ih_l{layer}'].detach().numpy()
        rnn2.layers[layer].bias_hh.data = rnn._parameters[f'bias_hh_l{layer}'].detach().numpy()

input2 = mytorch.randn(SEQUENCE_SIZE, BATCHES, INPUT_SIZE)
input2.data = input.detach().numpy()

h02 = mytorch.randn(LAYERS, BATCHES, HIDDEN_SIZE, requires_grad=True)
h02.data = h0.detach().numpy()

output2, hn2 = rnn2(input2, h02)

print(output2)

output2.backward()

print(h02.grad)