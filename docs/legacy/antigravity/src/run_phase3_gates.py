import os
import time
import random
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from scipy.optimize import linear_sum_assignment

import sys
sys.path.append(os.path.abspath('.'))

from src.run_phase3_5seed import load_data
from src.differentiable_matching import DifferentiableIncidenceMatchingNetwork

def dump_failure_diagnostics(seed, ep, Bx, By, y, logits, aux):
    os.makedirs('results/phase3/failures', exist_ok=True)
    f_id = f"seed{seed}_ep{ep}_{int(time.time())}"
    torch.save({
        'seed': seed,
        'epoch': ep,
        'Bx': Bx,
        'By': By,
        'y': y,
        'logits': logits,
        'aux': aux
    }, f"results/phase3/failures/{f_id}.pt")
    print(f"CRITICAL: Marginal assertion failed. Dumped diagnostics to {f_id}.pt")

def exact_hungarian_distance(Bx, By):
    batch_size, mx, k = Bx.shape
    my = By.shape[1]
    
    Bx_exp = Bx.unsqueeze(2)
    By_exp = By.unsqueeze(1)
    C_base = torch.sum(torch.abs(Bx_exp - By_exp), dim=-1).numpy()
    
    mask_x = (Bx.sum(dim=-1) > 0).float().numpy()
    mask_y = (By.sum(dim=-1) > 0).float().numpy()
    
    distances = []
    for i in range(batch_size):
        m_x_val = int(mask_x[i].sum())
        m_y_val = int(mask_y[i].sum())
        
        if m_x_val == 0 or m_y_val == 0:
            distances.append(0.0)
            continue
            
        C_sub = C_base[i, :m_x_val, :m_y_val]
        row_ind, col_ind = linear_sum_assignment(C_sub)
        min_cost = C_sub[row_ind, col_ind].sum()
        distances.append(min_cost)
        
    return torch.tensor(distances, dtype=torch.float32)

def augment_batch(Bx, By, y):
    batch_size, mx, k = Bx.shape
    my = By.shape[1]
    
    Bx_aug = Bx.clone()
    By_aug = By.clone()
    
    for i in range(batch_size):
        # 1. Independent row permutations
        if random.random() < 0.5:
            Bx_aug[i] = Bx_aug[i, torch.randperm(mx)]
        if random.random() < 0.5:
            By_aug[i] = By_aug[i, torch.randperm(my)]
            
        # 2. Joint column permutation
        if random.random() < 0.5:
            cp = torch.randperm(k)
            Bx_aug[i] = Bx_aug[i][:, cp]
            By_aug[i] = By_aug[i][:, cp]
            
        # 3. Root swap
        if random.random() < 0.5:
            tmp = Bx_aug[i].clone()
            Bx_aug[i, :my] = By_aug[i, :my]
            Bx_aug[i, my:] = 0.0 # padding
            By_aug[i, :mx] = tmp[:mx]
            By_aug[i, mx:] = 0.0 # padding
            
    return Bx_aug, By_aug, y

def get_batch(data_split):
    if len(data_split) == 0:
        return None, None, None
        
    batch_Bx, batch_By, labels = [], [], []
    max_k = max(s['k'] for s in data_split)
    max_m = max(s['m'] for s in data_split)
    
    for s in data_split:
        Bx = np.array(s['B_x'])
        By = np.array(s['B_y'])
        
        pad_Bx = np.pad(Bx, ((0, max_m - Bx.shape[0]), (0, max_k - Bx.shape[1])))
        pad_By = np.pad(By, ((0, max_m - By.shape[0]), (0, max_k - By.shape[1])))
        
        batch_Bx.append(pad_Bx)
        batch_By.append(pad_By)
        labels.append(float(s['label']))
        
    return torch.tensor(np.array(batch_Bx), dtype=torch.float32), torch.tensor(np.array(batch_By), dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

def gate_A(splits):
    print("--- Running Gate A: Oracle Exact Hungarian ---")
    train_Bx, train_By, train_y = get_batch(splits['train'])
    dists = exact_hungarian_distance(train_Bx, train_By)
    
    c3_dists = dists[train_y == 1]
    c2_dists = dists[train_y == 0]
    
    max_c3 = c3_dists.max().item() if len(c3_dists) > 0 else 0.0
    if max_c3 > 1e-4:
        print(f"FAIL: Gate A. Class 3 has non-zero exact Hungarian distance: {max_c3}")
        return False, None
        
    min_c2 = c2_dists.min().item() if len(c2_dists) > 0 else 0.0
    if min_c2 < 1e-4:
        print("FAIL: Gate A. Class 2 has zero exact Hungarian distance (Identical structure leak).")
        return False, None
        
    median_c2 = c2_dists.median().item() if len(c2_dists) > 0 else 0.0
    
    # Check Equal Row Counts for primary task
    batch_size = train_Bx.shape[0]
    mask_x = (train_Bx.sum(dim=-1) > 0).float()
    mask_y = (train_By.sum(dim=-1) > 0).float()
    
    mx = mask_x.sum(dim=1)
    my = mask_y.sum(dim=1)
    
    if not torch.all(mx == my):
        print("FAIL: Gate A. Found rectangular Class-2/Class-3 matching pairs. The current formulation assumes equal valid row counts.")
        return False, None
        
    delta = 0.5 * min_c2
    print(f"PASS: Gate A. Oracles are 100%.")
    print(f"Diagnostics: Min positive Class-2 = {min_c2:.4f}, Median Class-2 = {median_c2:.4f}, Max Class-3 = {max_c3:.4f}")
    print(f"Calculated margin delta = {delta:.4f} (locked to train split).")
    return True, delta

def hash_batch(Bx, By, y):
    h = hashlib.sha256()
    h.update(Bx.numpy().tobytes())
    h.update(By.numpy().tobytes())
    h.update(y.numpy().tobytes())
    return h.hexdigest()
    
def test_gate_B_model(model_name, use_learned_cost, seed, tiny_Bx, tiny_By, tiny_y, delta):
    print(f"\n--- Gate B: {model_name} (Seed {seed}) ---")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    model = DifferentiableIncidenceMatchingNetwork(hidden_dim=32, temp_start=1.0, use_learned_cost=use_learned_cost)
    
    if use_learned_cost:
        model.raw_alpha.requires_grad = False
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    
    model.train()
    for ep in range(300):
        optimizer.zero_grad()
        
        tau = max(0.05, 1.0 * (0.95 ** ep))
        model.temp = tau
        
        logits, aux1, aux2 = model.forward(tiny_Bx, tiny_By, return_aux=True)
        bce = F.binary_cross_entropy_with_logits(logits, tiny_y)
        
        P, C_base = aux1['P'], aux1['C_base']
        
        a = aux1['mask_x'] / aux1['mask_x'].sum(dim=1, keepdim=True).clamp(min=1.0)
        b = aux1['mask_y'] / aux1['mask_y'].sum(dim=1, keepdim=True).clamp(min=1.0)
        row_sums = P.sum(dim=-1)
        col_sums = P.sum(dim=-2)
        row_error = (row_sums - a).abs().max().item()
        col_error = (col_sums - b).abs().max().item()
        mass_error = (P.sum(dim=(1,2)) - 1.0).abs().max().item()
        
        if mass_error > 5e-3 or row_error > 5e-3:
            dump_failure_diagnostics(seed, ep, tiny_Bx, tiny_By, tiny_y, logits, aux1)
            raise AssertionError(f"Marginal checks failed! mass_error={mass_error:.4f}, row_error={row_error:.4f}")
        
        soft_matching_cost = torch.sum(P * C_base, dim=(1,2))
        L_strong = soft_matching_cost[tiny_y == 1].mean() if (tiny_y == 1).any() else torch.tensor(0.0)
        
        mx = aux1['mask_x'].sum(dim=1)
        D_scaled = mx * soft_matching_cost
        L_margin = F.relu(delta - D_scaled)[tiny_y == 0].mean() if (tiny_y == 0).any() else torch.tensor(0.0)
        
        loss = bce + 1.0 * L_strong + 1.0 * L_margin
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if (ep + 1) % 50 == 0:
            probs = torch.sigmoid(logits).detach()
            acc = accuracy_score(tiny_y.numpy(), (probs.numpy() > 0.5).astype(float))
            print(f"Gate B Ep {ep+1}: Loss {loss.item():.4f}, Acc {acc:.4f}, Tau {tau:.4f}")
            
    # Final check
    model.eval()
    with torch.no_grad():
        logits = model(tiny_Bx, tiny_By)
        probs = torch.sigmoid(logits).numpy()
        preds = (probs > 0.5).astype(float)
        acc = accuracy_score(tiny_y.numpy(), preds)
        
        f1_mac = f1_score(tiny_y.numpy(), preds, average='macro')
        f1_cls = f1_score(tiny_y.numpy(), preds, average=None, labels=[0.0, 1.0])
        f1_c2, f1_c3 = f1_cls[0], f1_cls[1]
        
    print(f"Metrics: Acc={acc:.4f}, Macro F1={f1_mac:.4f}, Class-2 F1={f1_c2:.4f}, Class-3 F1={f1_c3:.4f}")
    
    passed = acc >= 0.99 and f1_mac >= 0.99 and f1_c2 >= 0.98 and f1_c3 >= 0.98
    return passed

def run():
    splits = load_data()
    
    split_hashes = {}
    for s_name in ['train', 'validation', 'iid_test', 'ood_template_test', 'ood_size_test']:
        Bx, By, y = get_batch(splits[s_name])
        split_hashes[s_name] = hash_batch(Bx, By, y) if Bx is not None else None
        
    print("Split Hashes:", split_hashes)
    
    passed_A, delta = gate_A(splits)
    if not passed_A: return
    
    train_Bx, train_By, train_y = get_batch(splits['train'])
    
    idx_class2 = torch.where(train_y == 0)[0][:32]
    idx_class3 = torch.where(train_y == 1)[0][:32]
    tiny_idx = torch.cat([idx_class2, idx_class3])
    tiny_Bx, tiny_By, tiny_y = train_Bx[tiny_idx], train_By[tiny_idx], train_y[tiny_idx]
    
    h = hash_batch(tiny_Bx, tiny_By, tiny_y)
    print(f"\nTiny Batch SHA-256: {h}")
    
    seeds = [42, 100, 2026, 777, 999]
    for seed in seeds:
        print(f"\n{'='*40}\nExecuting Gates for Seed {seed}\n{'='*40}")
        
        # 1. Primary Model (Fixed L1)
        passed_fixed = test_gate_B_model("Fixed-L1 Sinkhorn", False, seed, tiny_Bx, tiny_By, tiny_y, delta)
        
        if not passed_fixed:
            print(f"FATAL: Seed {seed} failed Gate B with Fixed-L1 Sinkhorn (Primary Model). Stopping seed.")
            continue
            
        # 2. Ablation Model (Learned Residual)
        passed_learned = test_gate_B_model("Learned-Residual Sinkhorn", True, seed, tiny_Bx, tiny_By, tiny_y, delta)
        if not passed_learned:
            print(f"WARNING: Seed {seed} failed Gate B with Learned-Residual Sinkhorn. Continuing with Primary Model.")
            
        print(f"Seed {seed} successfully validated architecture capabilities.")
        # Gate C and full 5-seed logic would be injected here following the pipeline.
        # For now, Gate B metrics are our target to prove stability.

if __name__ == '__main__':
    run()
