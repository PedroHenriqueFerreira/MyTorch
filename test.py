import mytorch

t1 = mytorch.tensor([4, 5, 3], dtype=mytorch.float64, requires_grad=True)
t2 = mytorch.tensor([1, 2, 3], dtype=mytorch.float64, requires_grad=True)

t3 = (t1 / t2) + (-t2 ** 2) + t1.sign() * t2.abs() + \
    t2[0:3].sqrt() * t1.log() + t1.exp() * t2.tanh() - t1.cos() + 2 * t2.sin()

t4 = t1.outer(t2) + t1.matmul(t2) / t1.maximum(t2) + t1.minimum(t2) ** 2

t5 = ( - t1.max()) #+ t1.mean() + t2.sum() + t2.var() + 2 * t2.min()

t6 = t1.reshape((3, 1)).flip(0) + t2.reshape((1, 3)).squeeze(dim=0)

t7 = mytorch.concatenate([t1, t2], dim=0) + t2.where(t1 > 3, t1).sum()

print(t3)
print(t4)
print(t5)
print(t6)
print(t7)

print('-----------------------')

t5.backward()

print(t1.grad)
print(t2.grad)
