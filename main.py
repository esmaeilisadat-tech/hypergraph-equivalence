import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class HypergraphDataset(Dataset):
    def __init__(self, num_samples=1000, num_nodes=15):
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.data = []
        
        for _ in range(num_samples):
            c = np.random.randint(0, 3)
            H_matrix, label = self.generate_hypergraph(c)
            self.data.append((H_matrix, label))
            
    def generate_hypergraph(self, c):
        edges = []
        max_degree = 6
        num_neighbors = np.random.randint(2, 6)
        neighbors = np.random.choice(range(2, self.num_nodes), num_neighbors, replace=False).tolist()
        
        deg0 = 0
        deg1 = 0
        
        if c == 0:
            n0 = neighbors
            n1 = np.random.choice(range(2, self.num_nodes), num_neighbors, replace=False).tolist()
            if set(n0) == set(n1):
                n1 = [(x + 1) % self.num_nodes if (x + 1) % self.num_nodes not in [0, 1] else (x + 2) % self.num_nodes for x in n1]
            
            edges.append([0] + n0)
            edges.append([1] + n1)
            deg0 = 1
            deg1 = 1
            
        elif c == 1:
            edges.append([0] + neighbors)
            for neighbor in neighbors:
                edges.append([1, neighbor])
            deg0 = 1
            deg1 = num_neighbors
            
        elif c == 2:
            np.random.shuffle(neighbors)
            split_idx = len(neighbors) // 2
            group1 = neighbors[:split_idx]
            group2 = neighbors[split_idx:]
            
            if group1:
                edges.append([0] + group1)
                edges.append([1] + group1)
                deg0 += 1
                deg1 += 1
            if group2:
                edges.append([0] + group2)
                edges.append([1] + group2)
                deg0 += 1
                deg1 += 1
                
        # Add loops to balance degrees to max_degree
        for _ in range(max_degree - deg0):
            edges.append([0])
        for _ in range(max_degree - deg1):
            edges.append([1])
            
        # Add background edges
        for _ in range(10):
            edge_size = np.random.randint(2, 5)
            rand_edge = np.random.choice(range(2, self.num_nodes), edge_size, replace=False).tolist()
            edges.append(rand_edge)
            
        H = np.zeros((self.num_nodes, len(edges)), dtype=np.float32)
        for e_idx, edge in enumerate(edges):
            for node in edge:
                H[node, e_idx] = 1.0
                
        return H, c
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        H, label = self.data[idx]
        return torch.tensor(H), torch.tensor(label, dtype=torch.long)

class MLPBaseline(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.fc1 = nn.Linear(num_nodes * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)
        
    def forward(self, A):
        # A: (batch, num_nodes, num_nodes)
        x0 = A[:, 0, :]
        x1 = A[:, 1, :]
        x = torch.cat([x0, x1], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class IncidenceHGNN(nn.Module):
    def __init__(self, num_nodes, hidden_channels):
        super().__init__()
        self.num_nodes = num_nodes
        self.W1 = nn.Linear(3, hidden_channels)
        self.W2 = nn.Linear(hidden_channels, hidden_channels)
        self.W3 = nn.Linear(hidden_channels, hidden_channels)
        self.W4 = nn.Linear(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels * 2, 3)
        
    def forward(self, H):
        batch_size = H.size(0)
        X = torch.zeros((batch_size, self.num_nodes, 3), device=H.device)
        X[:, 0, 0] = 1.0
        X[:, 1, 1] = 1.0
        X[:, :, 2] = 1.0
        
        # Layer 1
        Y = torch.bmm(H.transpose(1, 2), X)
        Y = F.relu(self.W1(Y))
        X = torch.bmm(H, Y)
        X = F.relu(self.W2(X))
        
        # Layer 2
        Y = torch.bmm(H.transpose(1, 2), X)
        Y = F.relu(self.W3(Y))
        X = torch.bmm(H, Y)
        X = F.relu(self.W4(X))
        
        x0 = X[:, 0, :]
        x1 = X[:, 1, :]
        out = torch.cat([x0, x1], dim=-1)
        return self.fc(out)

def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for H, labels in dataloader:
        H = H.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        if isinstance(model, MLPBaseline):
            A = torch.bmm(H, H.transpose(1, 2))
            outputs = model(A)
        else:
            outputs = model(H)
            
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * H.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss / len(dataloader.dataset), acc, f1

def eval_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for H, labels in dataloader:
            H = H.to(device)
            labels = labels.to(device)
            
            if isinstance(model, MLPBaseline):
                A = torch.bmm(H, H.transpose(1, 2))
                outputs = model(A)
            else:
                outputs = model(H)
                
            loss = criterion(outputs, labels)
            total_loss += loss.item() * H.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss / len(dataloader.dataset), acc, f1

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Generating synthetic hypergraph dataset...")
    dataset = HypergraphDataset(num_samples=1000, num_nodes=15)
    train_size = 800
    test_size = 200
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print("\nTraining MLP Baseline...")
    mlp = MLPBaseline(num_nodes=15).to(device)
    optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, 21):
        loss, acc, f1 = train_model(mlp, train_loader, optimizer_mlp, criterion, device)
        if epoch % 5 == 0:
            print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")
            
    test_loss, test_acc, test_f1 = eval_model(mlp, test_loader, criterion, device)
    print(f"MLP Test -> Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f}")
    
    print("\nTraining Incidence-based HGNN...")
    hgnn = IncidenceHGNN(num_nodes=15, hidden_channels=32).to(device)
    optimizer_hgnn = torch.optim.Adam(hgnn.parameters(), lr=0.01)
    
    for epoch in range(1, 21):
        loss, acc, f1 = train_model(hgnn, train_loader, optimizer_hgnn, criterion, device)
        if epoch % 5 == 0:
            print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")
            
    test_loss, test_acc, test_f1 = eval_model(hgnn, test_loader, criterion, device)
    print(f"HGNN Test -> Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f}")

if __name__ == '__main__':
    main()
