import torch

t1 = torch.tensor([1, 2], dtype=torch.float32, requires_grad=True)
t2 = torch.tensor([3, 4], dtype=torch.float32, requires_grad=True)

t1 += t2