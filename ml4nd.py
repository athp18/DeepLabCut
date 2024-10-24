# PSet 1
import torch
# (a)

a = torch.arange(9).reshape(3, 3)

b = torch.arange(9).reshape(3, 3).t()

c = torch.arange(5).repeat(3)

d = torch.arange(5).repeat(3).reshape(3, 5)

e = torch.eye(4) + torch.diag(torch.full((3,), -2), diagonal=-1) + torch.diag(torch.full((3,), -2), diagonal=1)

f = torch.arange(3).reshape(1, 1, 1, 3)

a2 = 