import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

device = "cuda" if torch.cuda.is_available() else "cpu"

# n: the num of nodes
# T: the dim of embeddings

n = 10
T = 5
n_samples = 2
edge_prob = 0.4
C = 2

labels = torch.randint(0, C, (n,), device = device)
print(labels)

# adj_list_init: (n, 1)
adj_list = {i: [] for i in range(n)}


# adj_list: (n, avgly 0.4*n) Adjacency List
for i in range(n):
    for j in range(i + 1, n):
        if random.random() < edge_prob:
            adj_list[i].append(j)
            adj_list[j].append(i)

# X Feature Matrix: (n, T)
X = torch.rand((n, T), device=device)

# sample neighbors: (n, n_samples), random.choices will sample to n_samples if len(neighbours) < n_samples
def sample_neighbors(neighbors, k):
    if len(neighbors) == 0:
        return [0] * k
    return random.sample(neighbors, k) if len(neighbors) >= k else random.choices(neighbors, k=k)

# sampled_adj: (n, n_samples)
sampled_adj = torch.tensor(
    [sample_neighbors(adj_list[i], n_samples) for i in range(n)],
    dtype=torch.long,
    device=device
)

# print(f"sampled_adj: {sampled_adj}")

# hvk = sigma((hvk-1 || AGG(hu))W)
# X : (n, T), sampled_adj: (n, n_samples)
# aggregate: X[n_samlpes]

class GraphSAGELayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim*2, output_dim)
        
    def aggregate(self, X, sampled_adj):
        return torch.mean(X[sampled_adj], dim = 1)
        
    def forward(self, X, sampled_adj):
        aggegrated_neighbours = self.aggregate(X, sampled_adj)
        
        # hvk = sigma((hvk-1 || AGG(hu))W)
        hvk = torch.relu(self.linear(torch.cat([X, aggegrated_neighbours], dim = 1)))
        
        return hvk
        
        
class GraphSAGE(nn.Module):
    def __init__(self, T):
        super().__init__()
        
        self.layer1 = GraphSAGELayer(T, 8)
        self.layer2 = GraphSAGELayer(8, 4)
        
        self.classifier = nn.Linear(4, 2)
    
    def forward(self, X, sampled_adj):
        h = self.layer1(X, sampled_adj)
        h = self.layer2(h, sampled_adj)
        
        return self.classifier(h)


epochs = 100

model = GraphSAGE(T).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 0.01
)

criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    
    optimizer.zero_grad()
    
    logits = model(X, sampled_adj)
    
    loss = criterion(logits, labels)
    
    loss.backward()
    
    optimizer.step()
    
    if epoch % 10 == 0:
        print(
        epoch,
        loss.item()
    )   

model.eval()

with torch.no_grad():
    logits = model(X, sampled_adj)
    pred = torch.argmax(logits, dim = 1)

print(pred)

print(
f"""
confusion_matrix: 
{confusion_matrix(labels, pred)}
""")