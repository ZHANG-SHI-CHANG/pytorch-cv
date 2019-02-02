import torch
from torch import nn

a = nn.Conv2d(3, 16, 3, 1, 1)
b = nn.Conv2d(16, 4, 1, 1, groups=4)

num = torch.randn(2, 3, 10, 10)
print(b(a(num)).shape)
