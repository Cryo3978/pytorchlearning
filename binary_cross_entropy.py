import torch
import numpy

device = "cuda" if torch.cuda.is_available() else "cpu"

y_hat = torch.rand((10000,1), device = device)
y = torch.randint(0, 2, (10000,1), device = device).float()

def binary_cross_entropy(y_hat, y):
	return -(y*torch.log(y_hat)+(1-y)*torch.log(1-y_hat))

print(binary_cross_entropy(y_hat, y))
