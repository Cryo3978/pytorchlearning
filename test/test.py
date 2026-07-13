import torch
import numpy
from torch.nn.functional import softmax

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

a = torch.rand(10,512, device = device)

def test(a):
	return softmax(a, dim = 1)
print(test(a))