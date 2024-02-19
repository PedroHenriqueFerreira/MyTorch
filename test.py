from torch.nn import Sequential, Linear, Sigmoid, BCELoss, CrossEntropyLoss
from torch.optim import Adam

import torch

xor = Sequential(
    Linear(2, 2),
    Sigmoid(),
    Linear(2, 1),
    Sigmoid()
)

optimizer = Adam(xor.parameters())
loss_func = BCELoss()

x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
 
# Training loop
for i in range(1000000):
    optimizer.zero_grad()
    
    predict = xor(x)
    
    loss = loss_func(predict, y)
    
    loss.backward()
    
    optimizer.step()
    
    print(f'Result: {predict.data}, Epoch {i}, Loss {loss.data}')