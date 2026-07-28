import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import json

from src.binary_dataset import generate_matched_binary_dataset, split_dataset, extract_handcrafted_features, get_canonical_signature
from src.incidence_star_model import NestedDeepSets, PairConditionedBipartiteEncoder
from src.differentiable_matching import DifferentiableIncidenceMatchingNetwork
from src.oracles import exact_counter_oracle, hungarian_matching_oracle

def get_batch(dataset, max_m, max_k):
    batch_Bx = []
    batch_By = []
    labels = []
    for s in dataset:
        Bx = s['B_x']
        By = s['B_y']
        
        pad_Bx = np.pad(Bx, ((0, max_m - Bx.shape[0]), (0, max_k - Bx.shape[1])))
        pad_By = np.pad(By, ((0, max_m - By.shape[0]), (0, max_k - By.shape[1])))
        
        batch_Bx.append(pad_Bx)
        batch_By.append(pad_By)
        labels.append(float(s['label']))
        
    return torch.tensor(np.array(batch_Bx)), torch.tensor(np.array(batch_By)), torch.tensor(labels)

def train_and_eval(model_name, model, train_set, iid_test, ood_test, max_m, max_k, epochs=100, lr=0.005):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    train_Bx, train_By, train_y = get_batch(train_set, max_m, max_k)
    iid_Bx, iid_By, iid_y = get_batch(iid_test, max_m, max_k)
    ood_Bx, ood_By, ood_y = get_batch(ood_test, max_m, max_k)
    
    model.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        logits = model(train_Bx, train_By)
        loss = criterion(logits, train_y)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        train_logits = model(train_Bx, train_By)
        train_preds = (torch.sigmoid(train_logits) > 0.5).float()
        train_acc = accuracy_score(train_y.numpy(), train_preds.numpy())
        
        if len(iid_Bx) > 0:
            iid_logits = model(iid_Bx, iid_By)
            iid_preds = (torch.sigmoid(iid_logits) > 0.5).float()
            iid_acc = accuracy_score(iid_y.numpy(), iid_preds.numpy())
        else:
            iid_acc = 0.0
        
        ood_logits = model(ood_Bx, ood_By)
        ood_preds = (torch.sigmoid(ood_logits) > 0.5).float()
        ood_acc = accuracy_score(ood_y.numpy(), ood_preds.numpy())
        
        # Test Embeddings difference on OOD matched pairs
        # We assume OOD dataset contains matched pairs consecutively (since we grouped them)
        diffs = []
        for i in range(0, len(ood_Bx), 2):
            # Check if they are actually a pair
            if ood_y[i] != ood_y[i+1]:
                emb1 = model.get_embeddings(ood_Bx[i:i+1], ood_By[i:i+1])
                emb2 = model.get_embeddings(ood_Bx[i+1:i+2], ood_By[i+1:i+2])
                dist = torch.norm(emb1 - emb2).item()
                diffs.append(dist)
                
        avg_dist = np.mean(diffs) if diffs else 0.0
        
    return {
        "Train Acc": float(train_acc),
        "IID Acc": float(iid_acc),
        "OOD Acc": float(ood_acc),
        "Pair Embed Dist": float(avg_dist)
    }

def run_diagnostic():
    print("Generating matched dataset...")
    dataset = generate_matched_binary_dataset(num_pairs=500)
    
    print("Splitting dataset...")
    train_set, iid_val, iid_test, ood_test = split_dataset(dataset)
    print(f"Train: {len(train_set)}, IID Test: {len(iid_test)}, OOD Test: {len(ood_test)}")
    
    max_k = max([s['k'] for s in dataset])
    max_m = max([s['m'] for s in dataset])
    
    # Tiny overfit test
    tiny_train = train_set[:32]
    
    models = {
        "Model A: Nested DeepSets": NestedDeepSets(hidden_dim=32),
        "Model B: Pair-Cond Bipartite": PairConditionedBipartiteEncoder(hidden_dim=32),
        "Model C: Differentiable Matching": DifferentiableIncidenceMatchingNetwork(hidden_dim=32)
    }
    
    results = {}
    
    # 1. Oracle checks
    exact_acc = accuracy_score([s['label'] for s in ood_test], [exact_counter_oracle(s['B_x'], s['B_y']) for s in ood_test])
    hungarian_acc = accuracy_score([s['label'] for s in ood_test], [hungarian_matching_oracle(s['B_x'], s['B_y']) for s in ood_test])
    
    print(f"Exact Oracle OOD Acc: {exact_acc:.4f}")
    print(f"Hungarian Oracle OOD Acc: {hungarian_acc:.4f}")
    
    for name, model in models.items():
        print(f"\\nEvaluating {name}...")
        
        # Tiny Overfit Check
        tiny_res = train_and_eval(name, model, tiny_train, iid_test, ood_test, max_m, max_k, epochs=200, lr=0.01)
        print(f"Tiny Batch Overfit Train Acc: {tiny_res['Train Acc']:.4f}")
        
        # Full Train Check
        # Reset model
        if "Model A" in name:
            model = NestedDeepSets(hidden_dim=32)
        elif "Model B" in name:
            model = PairConditionedBipartiteEncoder(hidden_dim=32)
        else:
            model = DifferentiableIncidenceMatchingNetwork(hidden_dim=32)
            
        full_res = train_and_eval(name, model, train_set, iid_test, ood_test, max_m, max_k, epochs=150, lr=0.005)
        
        results[name] = {
            "Tiny Overfit Acc": tiny_res['Train Acc'],
            "Full Train Acc": full_res['Train Acc'],
            "IID Test Acc": full_res['IID Acc'],
            "OOD Test Acc": full_res['OOD Acc'],
            "Pair Embed Dist": full_res['Pair Embed Dist']
        }
        
        print(f"Full Train Acc: {full_res['Train Acc']:.4f}")
        print(f"IID Test Acc: {full_res['IID Acc']:.4f}")
        print(f"OOD Test Acc: {full_res['OOD Acc']:.4f}")
        print(f"Pair Embed Dist: {full_res['Pair Embed Dist']:.6f}")
        
    os.makedirs("results", exist_ok=True)
    with open("results/phase3_seed42_results.txt", "w") as f:
        f.write(f"Exact Oracle OOD Acc: {exact_acc:.4f}\\n")
        f.write(f"Hungarian Oracle OOD Acc: {hungarian_acc:.4f}\\n\\n")
        for name, metrics in results.items():
            f.write(f"--- {name} ---\\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.6f}\\n")
            f.write("\\n")
            
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    run_diagnostic()
