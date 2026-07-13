import torch
import numpy as np
import time

data = torch.rand(512, device = "cuda")

def l1_norm(data):
    start_l1 = time.perf_counter()
    res = data / sum(abs(x) for x in data)
    end_l1 = time.perf_counter()
    print(f'l1_time = {end_l1 - start_l1}') 
    return res
    

    
 
def l2_norm(data):
    start_l2 = time.perf_counter()
    res = data / (sum(x**2 for x in data)**0.5)
    end_l2 = time.perf_counter()
    print(f'l2_time = {end_l2 - start_l2}')
    return res
    
l1_norm(data)
l2_norm(data)
