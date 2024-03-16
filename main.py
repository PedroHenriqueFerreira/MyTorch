from mytorch import nn
import mytorch

rnn = nn.RNN(10, 20, 2)
input = mytorch.randn(5, 3, 10)
h0 = mytorch.randn(2, 3, 20)
output, hn = rnn(input, h0)

print(len(output), output[0].shape)