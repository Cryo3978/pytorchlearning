import torch
import numpy

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

a = torch.randn(10000, 1, device = device)

def sigmoid(a):
	a_sigmoid = 1 / (1+ torch.exp(a))
	return a_sigmoid

print(sigmoid(a))