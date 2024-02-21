import torch

torch.Tensor()

t1 = torch.tensor([-3., 1., 7.], dtype=torch.float32, requires_grad=True)
t2 = torch.tensor([4., 1., 6.], dtype=torch.float32, requires_grad=True)

t3 = t1 ** 2 + t2

grad = torch.tensor([1., 1., 1.], dtype=torch.float32, requires_grad=False)
grad2 = torch.tensor([1., 1., 1.], dtype=torch.float32, requires_grad=False)

print(t3)

t3.backward(grad, create_graph=True)
# t3.backward(grad, create_graph=True)

print('T3', t3)

print('-----------')

print(t1.grad)

g = t1.grad

t1.grad = None

g.backward(grad2)

print('-----------')

print(t1.grad)