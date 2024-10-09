import mytorch
from mytorch import nn as nn2

from torch import nn
import torch

INPUT_SIZE = 3 # OK
HIDDEN_SIZE = 5 # OK 
BATCHES = 5 # OK

lstm = nn.LSTMCell(INPUT_SIZE, HIDDEN_SIZE, True)

for a in lstm._parameters:
    print(a, '->', lstm._parameters[a].shape)
    
# Lotes, Entradas
input = torch.randn(BATCHES, INPUT_SIZE)

# Lotes, Escondidos
h0 = torch.randn(BATCHES, HIDDEN_SIZE, requires_grad=True)
c0 = torch.randn(BATCHES, HIDDEN_SIZE, requires_grad=True)

hx, cx = lstm.forward(input, (h0, c0))

print(hx)
print(cx)

hx.backward(torch.ones_like(hx))

print(h0.grad)

print('-' * 50)


lstm2 = nn2.LSTMCell(INPUT_SIZE, HIDDEN_SIZE, True)

lstm2.weight_ih.data = lstm._parameters[f'weight_ih'].detach().numpy()
lstm2.weight_hh.data = lstm._parameters[f'weight_hh'].detach().numpy()
lstm2.bias_ih.data = lstm._parameters[f'bias_ih'].detach().numpy()
lstm2.bias_hh.data = lstm._parameters[f'bias_hh'].detach().numpy()

input2 = mytorch.randn(BATCHES, INPUT_SIZE)
input2.data = input.detach().numpy()

h02 = mytorch.randn(BATCHES, HIDDEN_SIZE, requires_grad=True)
h02.data = h0.detach().numpy()
    
c02 = mytorch.randn(BATCHES, HIDDEN_SIZE, requires_grad=True)
c02.data = c0.detach().numpy()

hx2, cx2 = lstm2(input2, (h02, c02))

print(hx2)
print(cx2)

hx2.backward()

print(h02.grad)