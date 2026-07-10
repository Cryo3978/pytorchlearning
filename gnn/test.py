import torch
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

n_samples = 5

A = [[0 for _ in range(3)] for _ in range(5)]

for i in range(len(A)):
    for j in range(len(A[0])):
        A[i][j] = random.randint(0,100)

print(A)
for node in A:
    print(random.choices(node, k=n_samples))
    
A_prime = torch.tensor(
    A,
    dtype = torch.float32,
    device = device
)

print(f"A_prime: {A_prime}")