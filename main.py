import mytorch
from mytorch import nn as nn2

from torch import nn
import torch

rnn = nn.RNN(5, 6, 2)
input = torch.randn(5, 3, 5)
h0 = torch.randn(2, 3, 6, requires_grad=True)
output, hn = rnn(input, h0)

print(output)

output.backward(torch.ones_like(output))

# print(h0.grad)

print('-' * 50)


rnn2 = nn2.RNN(5, 6, 2)

rnn2.cells[0].weight_ih.data = rnn._parameters['weight_ih_l0'].detach().numpy()
rnn2.cells[0].weight_hh.data = rnn._parameters['weight_hh_l0'].detach().numpy()
rnn2.cells[0].bias_ih.data = rnn._parameters['bias_ih_l0'].detach().numpy()
rnn2.cells[0].bias_hh.data = rnn._parameters['bias_hh_l0'].detach().numpy()

rnn2.cells[1].weight_ih.data = rnn._parameters['weight_ih_l1'].detach().numpy()
rnn2.cells[1].weight_hh.data = rnn._parameters['weight_hh_l1'].detach().numpy()
rnn2.cells[1].bias_ih.data = rnn._parameters['bias_ih_l1'].detach().numpy()
rnn2.cells[1].bias_hh.data = rnn._parameters['bias_hh_l1'].detach().numpy()

input2 = mytorch.randn(5, 3, 5)

input2.data = input.detach().numpy()

h02 = mytorch.randn(2, 3, 6, requires_grad=True)

h02.data = h0.detach().numpy()

output2, hn2 = rnn2(input2, h02)

print(output2)

output2.backward()

# print(h02.grad)