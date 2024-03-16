from torch import nn
import torch

drop = nn.Dropout(0.5)

print(list(drop.named_parameters()))