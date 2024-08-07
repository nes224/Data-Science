import torch
import numpy as np 

# a = torch.ones(5)
# print(a)
# b = a.numpy()
# print(type(b))
"""
If the tensor is on the CPU and not the GPU then
both objects will share memory location so this means, *(torch, numpy)
that if we change one we will also change the other.
"""
# a.add_(1)
# print(a)
# print(b)