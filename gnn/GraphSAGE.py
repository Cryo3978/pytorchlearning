import torch.nn as nn
import torch
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

# n : num of nodes
# T : num of attributes for each node

n = 10
T = 5
n_samples = 2

edge_prob = 0.4
adj_list = []
for i in range(n):
    temp_list = []
    for j in range(n):
        if random.random() < edge_prob:
            temp_list.append(j)
    adj_list.append(temp_list) 
    
print(adj_list)
    
    
X = torch.rand((n,T), dtype = torch.float32, device = device)
labels = torch.randint(0,2, (n,), dtype = torch.float32, device = device)

def sampling_neighbours(adj_list_node, n_samples):
    neighbours = adj_list_node
    
    if len(neighbours) == 0:
        return []
    elif len(neighbours) >= n_samples:
        return random.sample(neighbours, n_samples)
    else:
        return random.choices(neighbours, k = n_samples)


sampled_adj_list = []
for i in range(n):
    sampled_adj_list.append(sampling_neighbours(adj_list[i], n_samples = n_samples))
    
print(f"sampled_adj_list: {sampled_adj_list}")
print(X[sampled_adj_list])