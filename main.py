from torch import tensor
from torch.optim import SGD
from torch.nn import ac

t1 = tensor([1., 2., 3.], requires_grad=True)

sgd = SGD([t1], lr=0.1)

for i in range(10):
    loss = t1.sum()
    loss.backward()
    sgd.step()
    sgd.zero_grad()
    print(t1)