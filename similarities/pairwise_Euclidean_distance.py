import torch

a = torch.randn(10, 512, device = "cuda")
b = torch.randn(10, 512, device = "cuda")

sum((a-b)**2)**0.5

def pairwise_Euclidean_distance(a,b):
    