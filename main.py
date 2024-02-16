from mytorch.nn import Sequential, Linear, ReLU, Sigmoid, BCELoss
from mytorch.optim import Adam

import mytorch as mt

nn = Sequential(
    Linear(2, 2),
    ReLU(),
    Linear(2, 2),
    ReLU(),
    Linear(2, 2),
    ReLU(),
    Linear(2, 1),
    Sigmoid(),
)

optimizer = Adam(nn.parameters(), lr=0.01)
loss = BCELoss()

x = mt.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=mt.float32)
y = mt.tensor([[0], [1], [1], [0]], dtype=mt.float32)
 
# Training loop
for i in range(1000000):
    optimizer.zero_grad()
    
    p = nn(x)
    
    l = loss(p, y)
    
    l.backward()
    
    optimizer.step()
    
    if i % 10 == 0:
        print(f'Epoch {i}, Loss {l.data}')