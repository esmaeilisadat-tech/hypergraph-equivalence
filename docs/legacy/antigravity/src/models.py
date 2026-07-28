import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBaseline(nn.Module):
    def __init__(self, num_nodes=20):
        super().__init__()
        # A_feat is concatenated rows x and y: 2 * num_nodes
        self.fc1 = nn.Linear(num_nodes * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)
        
    def forward(self, A_feat):
        x = F.relu(self.fc1(A_feat))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class IncidenceHGNN(nn.Module):
    def __init__(self, num_nodes=20, hidden_channels=32):
        super().__init__()
        self.num_nodes = num_nodes
        self.W1 = nn.Linear(3, hidden_channels)
        self.W2 = nn.Linear(hidden_channels, hidden_channels)
        self.W3 = nn.Linear(hidden_channels, hidden_channels)
        self.W4 = nn.Linear(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels * 2, 4)
        
    def forward(self, H, X):
        # H is (batch, num_nodes, max_edges)
        # X is (batch, num_nodes, 3)
        
        # Node to hyperedge: Y = H^T * X -> (batch, max_edges, 3)
        Y = torch.bmm(H.transpose(1, 2), X)
        Y = F.relu(self.W1(Y))
        
        # Hyperedge to node: X_new = H * Y -> (batch, num_nodes, hidden)
        X_new = torch.bmm(H, Y)
        X_new = F.relu(self.W2(X_new))
        
        # Layer 2
        Y = torch.bmm(H.transpose(1, 2), X_new)
        Y = F.relu(self.W3(Y))
        X_new = torch.bmm(H, Y)
        X_new = F.relu(self.W4(X_new))
        
        # Extract features for x and y
        # We know x and y have feature 0 and 1 = 1 in X
        # Instead of fixed indices 0 and 1, we can use dot product or gather
        # But since X has one-hot indicator, let's just do weighted sum
        ind_x = X[:, :, 0].unsqueeze(2) # (batch, num_nodes, 1)
        ind_y = X[:, :, 1].unsqueeze(2)
        
        x_feats = (X_new * ind_x).sum(dim=1) # (batch, hidden)
        y_feats = (X_new * ind_y).sum(dim=1)
        
        out = torch.cat([x_feats, y_feats], dim=-1)
        return self.fc(out)
