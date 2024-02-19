import torch
import mytorch

t1 = torch.tensor([3., 1., 7.], dtype=torch.float32, requires_grad=True)
t2 = torch.tensor([4., 1., 6.], dtype=torch.float32, requires_grad=True)

t3 = mytorch

print(t3)

t3.backward(torch.tensor([1., 1., 1.], dtype=torch.float32))

print(t1.grad)
print(t2.grad)