import torch
import time

a = torch.randn(10, 512, device = "cpu")

def softmax(a):
    start = time.perf_counter()
    
    exp_a = torch.exp(a)
    
    end = time.perf_counter()
    
    print(f'time_consumption:{end - start}')
    
    return exp_a / exp_a.sum(dim = 1, keepdim = True)
    

print(softmax(a))

def stable_softmax(x):
    x = x - x.max(dim = 1, keepdim = True).values
    
    exp_x = torch.exp(x)
    
    return exp_x / exp_x.sum(dim = 1, keepdim = True)
    
print(stable_softmax(a))

from torch.nn.functional import softmax as Softmax

print(torch.allclose(Softmax(a, dim = -1),stable_softmax(a),atol = 1e-6))