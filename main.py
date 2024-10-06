import mytorch
from mytorch import nn as nn2

from torch import nn
import torch

# Entrada, Escondidos, Camadas
rnn = nn.RNN(2, 3, 2, 'tanh', True, True, bidirectional=True)

# print([a for a in rnn._parameters])
for a in rnn._parameters:
    print(a, '->', rnn._parameters[a].shape)

# Momentos, Lotes, Entradas
input = torch.randn(2, 2, 2)

# Camadas, Lotes, Escondidos
h0 = torch.randn(2 * 2, 2, 3, requires_grad=True)

output, hn = rnn(input, h0)

print(output)

output.backward(torch.ones_like(output))

# print(h0.grad)

print('-' * 50)

rnn2 = nn2.RNN(2, 3, 2, 'tanh', True, True, bidirectional=True)

rnn2.cells[0].weight_ih.data = rnn._parameters['weight_ih_l0'].detach().numpy()
rnn2.cells[0].weight_hh.data = rnn._parameters['weight_hh_l0'].detach().numpy()
rnn2.cells[0].bias_ih.data = rnn._parameters['bias_ih_l0'].detach().numpy()
rnn2.cells[0].bias_hh.data = rnn._parameters['bias_hh_l0'].detach().numpy()
rnn2.cells_reverse[0].weight_ih.data = rnn._parameters['weight_ih_l0_reverse'].detach().numpy()
rnn2.cells_reverse[0].weight_hh.data = rnn._parameters['weight_hh_l0_reverse'].detach().numpy()
rnn2.cells_reverse[0].bias_ih.data = rnn._parameters['bias_ih_l0_reverse'].detach().numpy()
rnn2.cells_reverse[0].bias_hh.data = rnn._parameters['bias_hh_l0_reverse'].detach().numpy()

rnn2.cells[1].weight_ih.data = rnn._parameters['weight_ih_l1'].detach().numpy()
rnn2.cells[1].weight_hh.data = rnn._parameters['weight_hh_l1'].detach().numpy()
rnn2.cells[1].bias_ih.data = rnn._parameters['bias_ih_l1'].detach().numpy()
rnn2.cells[1].bias_hh.data = rnn._parameters['bias_hh_l1'].detach().numpy()
rnn2.cells_reverse[1].weight_ih.data = rnn._parameters['weight_ih_l1_reverse'].detach().numpy()
rnn2.cells_reverse[1].weight_hh.data = rnn._parameters['weight_hh_l1_reverse'].detach().numpy()
rnn2.cells_reverse[1].bias_ih.data = rnn._parameters['bias_ih_l1_reverse'].detach().numpy()
rnn2.cells_reverse[1].bias_hh.data = rnn._parameters['bias_hh_l1_reverse'].detach().numpy()

input2 = mytorch.randn(2, 2, 2)
input2.data = input.detach().numpy()

h02 = mytorch.randn(2 * 2, 2, 3, requires_grad=True)
h02.data = h0.detach().numpy()

output2, hn2 = rnn2(input2, h02)

print(output2)

output2.backward()

# print(h02.grad)