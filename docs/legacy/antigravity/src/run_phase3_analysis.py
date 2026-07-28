import os
import json
import torch
import pandas as pd
import numpy as np
import random
from scipy import stats
from sklearn.metrics import accuracy_score

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.run_phase3_5seed import load_data, get_batch, init_model

def get_accuracy(model, Bx, By, y):
    with torch.no_grad():
        logits = model(Bx, By)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
    return accuracy_score(y.numpy(), preds.numpy())

def run_sinkhorn_analysis(splits):
    # Analyze sensitivity of Sinkhorn temp and iters on the best model (seed 42)
    model = init_model('MatchingNetwork')
    chkpt = torch.load("results/phase3/checkpoints/MatchingNetwork_seed42.pt")
    model.load_state_dict(chkpt)
    model.eval()
    
    Bx, By, y = get_batch(splits['ood_template_test'])
    
    temps = [0.01, 0.05, 0.1, 0.5, 1.0]
    iters = [5, 10, 20, 50]
    
    results = []
    
    # Base L1 distance from the model
    with torch.no_grad():
        Rx = model.encode_rows(Bx)
        Ry = model.encode_rows(By)
        
        batch_size, mx, k = Bx.shape
        my = By.shape[1]
        
        Bx_exp = Bx.unsqueeze(2)
        By_exp = By.unsqueeze(1)
        C_raw = torch.sum(torch.abs(Bx_exp - By_exp), dim=-1)
        
        Rx_exp = Rx.unsqueeze(2).expand(batch_size, mx, my, model.hidden_dim)
        Ry_exp = Ry.unsqueeze(1).expand(batch_size, mx, my, model.hidden_dim)
        pair_feats = torch.cat([Rx_exp, Ry_exp], dim=-1)
        C_neural = model.cost_mlp(pair_feats).squeeze(-1)
        
        C = C_raw + C_neural
        
        # Calculate Hungarian cost for comparison
        from scipy.optimize import linear_sum_assignment
        hungarian_costs = []
        for i in range(batch_size):
            row_ind, col_ind = linear_sum_assignment(C[i].numpy())
            cost = C[i, row_ind, col_ind].sum().item()
            hungarian_costs.append(cost)
        hungarian_costs = np.array(hungarian_costs)
        
        mask_x = (Bx.sum(dim=-1) > 0).float()
        mask_y = (By.sum(dim=-1) > 0).float()
        
        results = []
        diagnostics = []
        
        for t in temps:
            for num_it in iters:
                model.temp = t
                model.num_iters = num_it
                
                # Manual sinkhorn using the model's new log-domain function
                P = model.sinkhorn(C, mask_x, mask_y, eps=t)
                
                soft_cost = torch.sum(P * C, dim=(1, 2)).numpy()
                approx_err = np.mean(np.abs(soft_cost - hungarian_costs))
                
                results.append({'Temperature': t, 'Iterations': num_it, 'Approximation_Error': approx_err})
                
                if t == 0.05 and num_it == 5:
                    # Save diagnostics for the default config
                    for i in range(batch_size):
                        m_x_val = mask_x[i].sum().item()
                        m_y_val = mask_y[i].sum().item()
                        if m_x_val == 0 or m_y_val == 0: continue
                        
                        P_i = P[i] # (mx, my)
                        C_i = C[i] # (mx, my)
                        
                        row_sums = P_i.sum(dim=1)
                        col_sums = P_i.sum(dim=0)
                        
                        expected_row = mask_x[i] / m_x_val
                        expected_col = mask_y[i] / m_y_val
                        
                        row_err = torch.abs(row_sums - expected_row).max().item()
                        col_err = torch.abs(col_sums - expected_col).max().item()
                        
                        P_valid = P_i[mask_x[i].bool()][:, mask_y[i].bool()]
                        entropy = -torch.sum(P_valid * torch.log(P_valid + 1e-12)).item()
                        
                        diagnostics.append({
                            'Sample_Idx': i,
                            'Finite_Cost': bool(torch.isfinite(C_i).all().item()),
                            'Finite_Assignment': bool(torch.isfinite(P_i).all().item()),
                            'Row_Marginal_Error': row_err,
                            'Column_Marginal_Error': col_err,
                            'Assignment_Entropy': entropy,
                            'Hungarian_Cost': hungarian_costs[i],
                            'Sinkhorn_Cost': soft_cost[i],
                            'Approximation_Error': abs(soft_cost[i] - hungarian_costs[i])
                        })
                
    df = pd.DataFrame(results)
    df.to_csv("results/phase3/sinkhorn_hungarian_comparison.csv", index=False)
    
    if len(diagnostics) > 0:
        pd.DataFrame(diagnostics).to_csv("results/phase3/sinkhorn_diagnostics.csv", index=False)

def run_permutation_tests(splits):
    # Test on seed 42 MatchingNetwork
    model = init_model('MatchingNetwork')
    chkpt = torch.load("results/phase3/checkpoints/MatchingNetwork_seed42.pt")
    model.load_state_dict(chkpt)
    model.eval()
    
    Bx, By, y = get_batch(splits['iid_test'])
    
    with torch.no_grad():
        base_logits = model(Bx, By)
        base_probs = torch.sigmoid(base_logits)
        
    results = []
    
    # 1. Row permutation Bx
    batch_size, mx, k = Bx.shape
    my = By.shape[1]
    
    Bx_perm = Bx.clone()
    for i in range(batch_size):
        rp = torch.randperm(mx)
        Bx_perm[i] = Bx[i, rp]
        
    with torch.no_grad():
        logits_perm = model(Bx_perm, By)
        probs_perm = torch.sigmoid(logits_perm)
        # Test logits and probs
        consistency_probs = torch.allclose(base_probs, probs_perm, atol=1e-6)
        consistency_logits = torch.allclose(base_logits, logits_perm, atol=1e-6)
        results.append({'Test': 'Row Permutation Bx', 'Consistent': consistency_probs and consistency_logits})
        
    # 2. Swap X and Y
    with torch.no_grad():
        logits_swap = model(By, Bx)
        probs_swap = torch.sigmoid(logits_swap)
        consistency_probs = torch.allclose(base_probs, probs_swap, atol=1e-6)
        consistency_logits = torch.allclose(base_logits, logits_swap, atol=1e-6)
        
        # Native vs Forced asymmetry
        base_dir = model.forward_direction(Bx, By)
        swap_dir = model.forward_direction(By, Bx)
        native_asymmetry = torch.abs(base_dir - swap_dir).mean().item()
        final_asymmetry = torch.abs(base_logits - logits_swap).mean().item()
        
        results.append({'Test': 'Swap Bx and By', 'Consistent': consistency_probs and consistency_logits, 'Native_Asym': native_asymmetry, 'Final_Asym': final_asymmetry})
        
        # Sinkhorn Transpose Check
        mask_x = (Bx.sum(dim=-1) > 0).float()
        mask_y = (By.sum(dim=-1) > 0).float()
        
        Bx_exp = Bx.unsqueeze(2)
        By_exp = By.unsqueeze(1)
        C_raw_x = torch.sum(torch.abs(Bx_exp - By_exp), dim=-1)
        Rx = model.encode_rows(Bx)
        Ry = model.encode_rows(By)
        Rx_exp = Rx.unsqueeze(2).expand(batch_size, mx, my, model.hidden_dim)
        Ry_exp = Ry.unsqueeze(1).expand(batch_size, mx, my, model.hidden_dim)
        C_neural_x = model.cost_mlp(torch.cat([Rx_exp, Ry_exp], dim=-1)).squeeze(-1)
        C_x = C_raw_x + C_neural_x
        
        C_raw_y = torch.sum(torch.abs(By.unsqueeze(2) - Bx.unsqueeze(1)), dim=-1)
        C_neural_y = model.cost_mlp(torch.cat([Ry.unsqueeze(2).expand(batch_size, my, mx, model.hidden_dim), Rx.unsqueeze(1).expand(batch_size, my, mx, model.hidden_dim)], dim=-1)).squeeze(-1)
        C_y = C_raw_y + C_neural_y
        
        P_x = model.sinkhorn(C_x, mask_x, mask_y, eps=model.temp)
        P_y = model.sinkhorn(C_y, mask_y, mask_x, eps=model.temp)
        
        P_x_transposed = P_x.transpose(1, 2)
        transpose_error = torch.abs(P_x_transposed - P_y).mean().item()
        results.append({'Test': 'Sinkhorn P(By,Bx) == P(Bx,By)^T', 'Consistent': transpose_error < 1e-4, 'Error': transpose_error})
        
    # 3. Joint column permutation
    Bx_col = Bx.clone()
    By_col = By.clone()
    cp = torch.randperm(k)
    for i in range(batch_size):
        Bx_col[i] = Bx[i][:, cp]
        By_col[i] = By[i][:, cp]
        
    with torch.no_grad():
        logits_col = model(Bx_col, By_col)
        probs_col = torch.sigmoid(logits_col)
        consistency_probs = torch.allclose(base_probs, probs_col, atol=1e-6)
        consistency_logits = torch.allclose(base_logits, logits_col, atol=1e-6)
        results.append({'Test': 'Joint Column Permutation', 'Consistent': consistency_probs and consistency_logits})
        
    pd.DataFrame(results).to_csv("results/phase3/permutation_tests.csv", index=False)

def run_bootstrap_ci():
    # Bootstrap CI between MatchingNetwork and BipartiteEncoder on ood_template_test
    df = pd.read_csv("results/phase3/per_sample_predictions.csv")
    
    # We want to do this for a specific seed or aggregate? Let's do Seed 42
    df_seed = df[(df['Seed'] == 42) & (df['Split'] == 'ood_template_test')]
    
    mn_preds = df_seed[df_seed['Model'] == 'MatchingNetwork']
    bp_preds = df_seed[df_seed['Model'] == 'BipartiteEncoder']
    
    # Sort by Pair ID to align
    mn_preds = mn_preds.sort_values('Matched_Pair_ID')
    bp_preds = bp_preds.sort_values('Matched_Pair_ID')
    
    y_true = mn_preds['True_Label'].values
    y_pred_mn = (mn_preds['Prob'].values > 0.5).astype(float)
    y_pred_bp = (bp_preds['Prob'].values > 0.5).astype(float)
    
    n_bootstraps = 1000
    diffs = []
    
    np.random.seed(42)
    for _ in range(n_bootstraps):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        acc_mn = accuracy_score(y_true[idx], y_pred_mn[idx])
        acc_bp = accuracy_score(y_true[idx], y_pred_bp[idx])
        diffs.append(acc_mn - acc_bp)
        
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    
    with open("results/phase3/bootstrap_comparisons.csv", "w") as f:
        f.write(f"ModelA,ModelB,MeanDiff,CI_2.5,CI_97.5\n")
        f.write(f"MatchingNetwork,BipartiteEncoder,{np.mean(diffs)},{ci_lower},{ci_upper}\n")

if __name__ == "__main__":
    splits = load_data()
    run_sinkhorn_analysis(splits)
    run_permutation_tests(splits)
    try:
        run_bootstrap_ci()
    except Exception as e:
        print("Bootstrap failed, maybe predictions not saved yet.", e)
