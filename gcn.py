import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

# n is the num of nodes, and T is the num of features

n = 5
T = 3
epochs = 100

# A is the Adjacency Matrix, the size is R^n*n  

A = torch.randint(low=0, high=2, size=(n, n), device = device)

I = torch.eye(A.size(0))

A_tilde = A+I

D = torch.sum(A_tilde, dim=1)

print(A_tilde)
print(D)

D_inv_sqrt = torch.diag(D.pow(-0.5))
print(D_inv_sqrt)

A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt

X = torch.rand((n, T), device = device)

class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        self.W = nn.Linear(
            input_dim,
            output_dim
        )
        
    def forward(self, X, A_hat):
        X = A_hat @ X
        
        X = self.W(X)
        
        return torch.relu(X)

class GCN(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.layer1=GCNLayer(3,4)
        self.layer2=GCNLayer(4,2)
        
    def forward(self, X, A):
        
        X = self.layer1(X,A)
        X = self.layer2(X,A)
        
        return X
        
model = GCN()

Z = model(X, A_hat)

print(Z)

labels = torch.randint(low = 0, high = 2, size = (n, 1), device = device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 0.01
)

for epoch in range(epochs):