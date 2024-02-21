import mytorch

a = mytorch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)

b = a.split((2,))

print(b)

c = b[1]

print(c)

grad = mytorch.ones(c.shape, requires_grad=True)

c.backward(grad)

grad2 = a.grad

a.grad = 0

gra = mytorch.ones(a.shape, requires_grad=True)

grad2.backward(gra)

print(b)

print(a.grad)