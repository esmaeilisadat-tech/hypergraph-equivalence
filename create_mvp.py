import json

notebook = {
 "cells": [],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

def add_code_cell(code):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    })

def add_md_cell(md):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in md.strip().split("\n")]
    })

add_md_cell("# MVP: Hypergraph Neural Network for Detecting Structural Equivalence")
add_code_cell("""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from fractions import Fraction
import pandas as pd
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # Required for deterministic behavior
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
""")

add_md_cell("## Mathematical Definitions (Ground Truth)")
add_code_cell("""
def get_neighbors(H_edges, x):
    n = set()
    for e in H_edges:
        if x in e:
            for z in e:
                if z != x:
                    n.add(z)
    return n

def ee_measure(H_edges, x):
    dx = sum(1 for e in H_edges if x in e)
    if dx == 0: return {}
    mu = {}
    for e in H_edges:
        if x in e:
            size = len(e)
            if size > 1:
                weight = Fraction(1, size - 1)
                for z in e:
                    if z != x:
                        if z not in mu:
                            mu[z] = Fraction(0)
                        mu[z] += weight / dx
    return mu

def get_incident_neighbor_sets(H_edges, x):
    res = []
    for e in H_edges:
        if x in e:
            s = set(e)
            s.remove(x)
            res.append(frozenset(s))
    return sorted(res, key=lambda s: (len(s), sorted(list(s))))

def strongly_structurally_equivalent(H_edges, x, y):
    sx = get_incident_neighbor_sets(H_edges, x)
    sy = get_incident_neighbor_sets(H_edges, y)
    return Counter(sx) == Counter(sy)

def classify_pair(H_edges, x, y):
    nx = get_neighbors(H_edges, x)
    ny = get_neighbors(H_edges, y)
    if nx != ny:
        return 0
    mu_x = ee_measure(H_edges, x)
    mu_y = ee_measure(H_edges, y)
    if mu_x != mu_y:
        return 1
    if not strongly_structurally_equivalent(H_edges, x, y):
        return 2
    return 3
""")

add_md_cell("## Dataset Generation")
add_code_cell("""
def _generate_hypergraph(c, num_nodes=20):
    edges = []
    x, y = 0, 1
    
    if c == 0: # Not equivalent
        num_n = random.randint(2, 4)
        nx = random.sample(range(2, num_nodes), num_n)
        ny = random.sample(range(2, num_nodes), num_n)
        if set(nx) == set(ny):
            ny[0] = (ny[0] + 1) % num_nodes
            if ny[0] in [0, 1]: ny[0] = 2
        edges.append([x] + nx)
        edges.append([y] + ny)
        
    elif c == 1: # Weak equivalent (N(x)==N(y), mu_x != mu_y)
        num_n = random.randint(3, 5)
        n_all = random.sample(range(2, num_nodes), num_n)
        if random.random() < 0.5:
            edges.append([x] + n_all)
            edges.append([y] + n_all[:2])
            edges.append([y] + n_all[2:])
        else:
            edges.append([x] + n_all[:2])
            edges.append([x] + n_all[2:])
            edges.append([y, n_all[0]])
            edges.append([y] + n_all[1:])
            
    elif c == 2: # EE-measure equivalent (not strong)
        num_n = random.randint(3, 4)
        n_all = random.sample(range(2, num_nodes), num_n)
        if random.random() < 0.5:
            edges.append([x] + n_all)
            for z in n_all:
                edges.append([y, z])
        else:
            n_all = random.sample(range(2, num_nodes), 3)
            a, b, c_n = n_all
            edges.append([x, a, b])
            edges.append([x, b, c_n])
            edges.append([x, a, c_n])
            edges.append([y, a])
            edges.append([y, b])
            edges.append([y, c_n])
            
    elif c == 3: # Strong equivalent
        num_n = random.randint(3, 5)
        n_all = random.sample(range(2, num_nodes), num_n)
        if random.random() < 0.5:
            edges.append([x] + n_all)
            edges.append([y] + n_all)
        else:
            edges.append([x] + n_all[:2])
            edges.append([x] + n_all[2:])
            edges.append([y] + n_all[:2])
            edges.append([y] + n_all[2:])
            
    # Add random background edges
    for _ in range(10):
        sz = random.randint(2, 5)
        bg = random.sample(range(2, num_nodes), sz)
        edges.append(bg)
        
    actual_c = classify_pair(edges, x, y)
    
    max_edges = 25
    H = np.zeros((num_nodes, max_edges), dtype=np.float32)
    for e_idx, e in enumerate(edges):
        for v in e:
            H[v, e_idx] = 1.0
            
    return H, actual_c

def generate_for_class(target_class, num_nodes=20):
    for _ in range(100):
        H, actual_class = _generate_hypergraph(target_class, num_nodes)
        if actual_class == target_class:
            return H, actual_class
    raise RuntimeError(f"Could not generate a valid sample for class {target_class}")

class HGDataset(Dataset):
    def __init__(self, n=1500):
        self.data = []
        counts = {0:0, 1:0, 2:0, 3:0}
        for _ in range(n):
            c = random.randint(0, 3)
            H, actual = generate_for_class(c)
            self.data.append((H, actual))
            counts[actual] += 1
        print("Total Dataset class counts:", counts)
        self.counts = counts
        
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0]), torch.tensor(self.data[idx][1])

print("Generating 1500 hypergraphs...")
dataset = HGDataset(1500)
train_size = 1000
test_size = 500

split_generator = torch.Generator().manual_seed(42)
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size], generator=split_generator
)

def print_split_counts(subset, name):
    counts = Counter([dataset.data[i][1] for i in subset.indices])
    print(f"{name} split class counts:", dict(counts))

print_split_counts(train_dataset, "Train")
print_split_counts(test_dataset, "Test")

# IMPORTANT: we must set workers_init_fn if using multiple workers to ensure determinism
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
""")

add_md_cell("## Models")
add_code_cell("""
class MLPBaseline(nn.Module):
    def __init__(self, num_nodes=20):
        super().__init__()
        self.fc1 = nn.Linear(num_nodes * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)
        
    def forward(self, A):
        x0 = A[:, 0, :]
        x1 = A[:, 1, :]
        x = torch.cat([x0, x1], dim=1)
        x = F.relu(self.fc1(x))
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
        
    def forward(self, H):
        batch_size = H.size(0)
        X = torch.zeros((batch_size, self.num_nodes, 3), device=H.device)
        X[:, 0, 0] = 1.0
        X[:, 1, 1] = 1.0
        X[:, :, 2] = 1.0
        
        Y = torch.bmm(H.transpose(1, 2), X)
        Y = F.relu(self.W1(Y))
        X = torch.bmm(H, Y)
        X = F.relu(self.W2(X))
        
        Y = torch.bmm(H.transpose(1, 2), X)
        Y = F.relu(self.W3(Y))
        X = torch.bmm(H, Y)
        X = F.relu(self.W4(X))
        
        x0 = X[:, 0, :]
        x1 = X[:, 1, :]
        out = torch.cat([x0, x1], dim=-1)
        return self.fc(out)
""")

add_md_cell("## Training")
add_code_cell("""
def train_and_eval(model, train_loader, test_loader, device, epochs=30):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for H, labels in train_loader:
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
            
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for H, labels in test_loader:
            H = H.to(device)
            if isinstance(model, MLPBaseline):
                A = torch.bmm(H, H.transpose(1, 2))
                outputs = model(A)
            else:
                outputs = model(H)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    f1_strong = f1_score(all_labels, all_preds, labels=[3], average='macro')
    
    print("\\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    
    return acc, f1, f1_strong

print("========================================")
print("Training MLP...")
mlp = MLPBaseline().to(device)
mlp_acc, mlp_f1, mlp_strong_f1 = train_and_eval(mlp, train_loader, test_loader, device, epochs=30)

print("========================================")
print("Training Incidence HGNN...")
hgnn = IncidenceHGNN().to(device)
hgnn_acc, hgnn_f1, hgnn_strong_f1 = train_and_eval(hgnn, train_loader, test_loader, device, epochs=30)
""")

add_md_cell("## Results Table")
add_code_cell("""
results = [
    {
        "Model": "MLP Baseline", 
        "Accuracy": mlp_acc,
        "Macro F1": mlp_f1, 
        "Strong equivalence F1": mlp_strong_f1
    },
    {
        "Model": "Incidence Hypergraph GNN", 
        "Accuracy": hgnn_acc,
        "Macro F1": hgnn_f1, 
        "Strong equivalence F1": hgnn_strong_f1
    }
]

df = pd.DataFrame(results)
df = df.round(4)
print("\\nFinal Comparison:\\n")
print(df.to_markdown(index=False))
""")

with open('MVP.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)
