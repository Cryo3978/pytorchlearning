import torch
import torch.nn.functional as F

num_q = 10
num_d = 100
Q = torch.randn(num_q, 512, device = "cuda")
D = torch.randn(num_d, 512, device = "cuda")

def pairwise_cosine_similarity(Q,D):
    Q_norm = F.normalize(Q, p=2, dim = 1)
    D_norm = F.normalize(D, p=2, dim = 1)
    
    res = Q_norm @ D_norm.T
    return res

print(torch.allclose(pairwise_cosine_similarity(Q,D), pairwise_cosine_similarity(D,Q).T,atol = 1e-6))