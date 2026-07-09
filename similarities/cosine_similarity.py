import torch
import random
import numpy as np
import time

# device = "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

dim = 100000
print(f"dim:{dim}")

random.seed(42)

a = [random.random() for i in range(dim)]
b = [random.random() for i in range(dim)]

def cosine_similarity(a,b) -> float:
    if len(a) != len(b):
        raise ValueError("a and b has different dims")
    else:
        a_data = torch.tensor(a).to(device)
        b_data = torch.tensor(b).to(device)
        ab_product = sum(x*y for x,y in zip(a_data,b_data))
        ab_sqrt = sum(x**2 for x in a_data)**0.5 * sum(y**2 for y in b_data)**0.5
        cs = ab_product/ab_sqrt
    return cs

start = time.perf_counter()
print(cosine_similarity(a,b))
end = time.perf_counter()

print(end - start)