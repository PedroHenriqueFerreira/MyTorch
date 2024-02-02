from torch import tensor, optim
from tensor import Tensor

t1 = tensor([1., 2., 3.], requires_grad=True)
t2 = Tensor([1., 2., 3.], requires_grad=True)

t1.mean().backward()
t2.mean().backward()

print(t1.grad)
print(t2.grad)