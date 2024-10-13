import torch
import mytorch

# t = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, requires_grad=True)

# y = torch.tile(t, (2, 2))

# print(y)

# y.backward(torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2], [1.3, 1.4, 1.5, 16]]))

# print(t.grad)

# print(t.repeat(3))

t2 = mytorch.tensor([[1, 2], [3, 4]], dtype=mytorch.float32, requires_grad=True)

print(t2.repeat(2, axis=1))