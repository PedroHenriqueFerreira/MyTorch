import torch
from aa import Tensor

v1 = torch.tensor([1., 2.], requires_grad=True)
m1 = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)

y1 = m1 @ v1

v2 = Tensor([1., 2.], requires_grad=True)
m2 = Tensor([[1., 2.], [3., 4.]], requires_grad=True)

y2 = m2 @ v2

y1.backward(torch.tensor([1., 1.]))
y2.backward([1., 1.])

print('v1 grad: ', v1.grad)
print('m1 grad: ', m1.grad)

print('-------------')

print('v2 grad: ', v2.grad)
print('m2 grad: ', m2.grad)