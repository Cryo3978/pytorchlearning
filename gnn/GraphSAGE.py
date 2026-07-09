import torch
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

# n: the num of nodes
# T: the dim of embeddings

n = 10
T = 5
n_samples = 2
edge_prob = 0.4

adj_list = {i: [] for i in range(n)}

for i in range(n):
    for j in range(i + 1, n):
        if random.random() < edge_prob:
            adj_list[i].append(j)
            adj_list[j].append(i)

X = torch.rand((n, T), device=device)

def sample_neighbors(neighbors, k):
    if len(neighbors) == 0:
        return [0] * k
    return random.sample(neighbors, k) if len(neighbors) >= k else random.choices(neighbors, k=k)

sampled_adj = torch.tensor(
    [sample_neighbors(adj_list[i], n_samples) for i in range(n)],
    dtype=torch.long,
    device=device
)

neighbor_emb = X[sampled_adj].mean(dim=1)

combined = torch.cat([X, neighbor_emb], dim=1)

print(adj_list)
print(sampled_adj.shape)
print(neighbor_emb.shape)
print(combined.shape)