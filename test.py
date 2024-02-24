import torch

t1 = torch.tensor([1, 2, 3], dtype=torch.float32, requires_grad=True)
t2 = torch.tensor([3, 4], dtype=torch.float32, requires_grad=False)

t3 = t1[0:2] ** t2

print(t3)
print(t3.grad_fn(torch.ones(t3.shape)))

grad = torch.ones(t3.shape, requires_grad=True)

t3.backward(grad, create_graph=True)

print(t1.grad)

t1grad = t1.grad

t1.grad = None

t1grad.backward(torch.ones(t1grad.shape))

print(t1.grad)

print('--------------------------------')

# raise NotImplementedError

import mytorch

m1 = mytorch.Tensor([1, 2, 3], dtype=mytorch.float32, requires_grad=True)
m2 = mytorch.Tensor([3, 4], dtype=mytorch.float32, requires_grad=False)

m3 = m1[0:2] ** m2

print(m3)
print(m3.grad_fn(torch.ones(m3.shape)))

grad = mytorch.ones(m3.shape, requires_grad=True)

m3.backward(grad)

print(m1.grad)

m1grad = m1.grad

m1.grad = None

m1grad.backward(mytorch.ones(m1grad.shape))

print(m1.grad)