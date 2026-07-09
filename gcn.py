import torch
import torch.nn as nn
from sklearn.metrics import confusion_martix

device = "cuda" if torch.cuda.is_available() else "cpu"

# n is the num of nodes, and T is the num of features

n = 5
T = 3
epochs = 100

# A is the Adjacency Matrix, the size is R^n*n  

A = torch.randint(low=0, high=2, size=(n, n), dtype = torch.float32, device = device)

A.fill_diagonal_(0)

I = torch.eye(A.size(0),device = device)

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

    # for a layer, H(l+1) = σ(A_hat @ Hl @ W)
    # where A_hat (n, n), H0 = X, and the size of X is (n, T), W is the paras to train, the size of it is (dim_input, dim_output)
    # The size of W depends on the task
    
    def forward(self, X, A_hat):
        X = A_hat @ X
        
        X = self.W(X)
        
        return torch.relu(X)

class GCN(nn.Module):
    
    def __init__(self, T):
        super().__init__()
        
        # It's a 2-layer GCN structure.
        
        self.layer1=GCNLayer(T,4)
        self.layer2=GCNLayer(4,2)
        
        self.classifier = nn.Linear(2,2)
        
    def forward(self, X, A_hat):
        
        X = self.layer1(X,A_hat)
        X = self.layer2(X,A_hat)
        
        pred = self.classifier(X)
        
        return pred
        
model = GCN(T).to(device)

Z = model(X, A_hat)

print(Z)

labels = torch.randint(low = 0, high = 2, size = (n, ), device = device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 0.01
)

for epoch in range(epochs):
    optimizer.zero_grad()
    
    pred = model(X, A_hat)
    
    loss = criterion(pred, labels)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 ==0:
        pred_class = torch.argmax(pred, dim = 1)
        print(f"epoch: {epoch}")
        print(f"loss:{loss.item()}")
        
        print(f"CM:{confusion_martix(pred_class, labels)}")