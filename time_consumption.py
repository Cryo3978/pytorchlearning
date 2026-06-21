import torch 
import numpy as np
import time

start_cuda = time.perf_counter()

x = torch.rand(8192,8192, device = "cuda")
y = torch.rand(8192,8192, device = "cuda")

x @ y

end_cuda = time.perf_counter()

print(f'time_consumption of cuda = {end_cuda - start_cuda}')



start_cpu = time.perf_counter()

x = torch.rand(8192,8192, device = "cpu")
y = torch.rand(8192,8192, device = "cpu")

x @ y

end_cpu = time.perf_counter()

print(f'time_consumption of cpu = {end_cpu - start_cpu}')