import torch

pred = torch.randn(10, 100000, device = "cuda")
logits = torch.randint(0, 100000, (10,), device = "cuda")

def cross_entropy_loss(a):
    prob = pred[torch.arange(10), true]
    