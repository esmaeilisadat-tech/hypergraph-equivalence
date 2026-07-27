import os
import json
import torch
import numpy as np
from tqdm import tqdm

from src.run_phase3_final import load_data, get_batch
from src.config_manager import load_config
from src.differentiable_matching import DifferentiableIncidenceMatchingNetwork

def compute_residuals(P, mask_x, mask_y):
    row_sum = P.sum(dim=2)
    col_sum = P.sum(dim=1)
    
    # Target row mass is 1.0 for valid rows (because Sinkhorn normalizes rows)
    # Wait, Sinkhorn normalizes rows to 1.0? 
    # Let's check: u = 1 / (K v), so u (K v) = 1. So row sum is 1.0 for valid rows.
    # What about col sum? It depends on the number of valid rows and cols!
    # Let's just check if row_sum matches mask_x exactly.
    row_residual = torch.abs(row_sum - mask_x).max().item()
    
    # Total mass should equal number of valid rows
    total_mass = P.sum(dim=(1,2))
    expected_mass = mask_x.sum(dim=1)
    mass_residual = torch.abs(total_mass - expected_mass).max().item()
    
    # P must be non-negative
    min_p = P.min().item()
    
    # Check NaN/Inf
    is_finite = torch.isfinite(P).all().item()
    
    return row_residual, mass_residual, min_p, is_finite

def audit_numerics():
    cfg = load_config("MatchingNetwork")
    seeds = [42, 100, 2026, 777, 999]
    splits = load_data()
    audit_splits = ['iid_test', 'ood_template_test', 'ood_size_test']
    
    results = {
        "num_audited": 0,
        "num_failures": 0,
        "mean_error": 0.0,
        "max_error": 0.0,
        "worst_sample_id": None,
        "worst_seed": None,
        "worst_split": None,
        "failed_quantity": None,
        "p95_error": 0.0,
        "p99_error": 0.0
    }
    
    all_errors = []
    worst_err = -1.0
    
    for seed in seeds:
        print(f"Auditing seed {seed}...")
        model = DifferentiableIncidenceMatchingNetwork(cfg)
        chkpt_path = f"results/phase3/checkpoints/MatchingNetwork_seed{seed}.pt"
        if not os.path.exists(chkpt_path):
            print(f"Skipping seed {seed}, checkpoint not found.")
            continue
            
        model.load_state_dict(torch.load(chkpt_path))
        model.eval()
        model.eval_iters = 500
        model.double() # Ensure float64
        
        for split_name in audit_splits:
            Bx, By, y = get_batch(splits[split_name])
            if Bx is None: continue
            
            Bx = Bx.double()
            By = By.double()
            
            bs = 32
            for i in range(0, Bx.shape[0], bs):
                bx_batch = Bx[i:i+bs]
                by_batch = By[i:i+bs]
                
                with torch.no_grad():
                    l1, aux1 = model.forward_direction(bx_batch, by_batch, return_aux=True)
                    l2, aux2 = model.forward_direction(by_batch, bx_batch, return_aux=True)
                    
                    logit_xy = model(bx_batch, by_batch)
                    logit_yx = model(by_batch, bx_batch)
                    prob_xy = torch.sigmoid(logit_xy)
                    prob_yx = torch.sigmoid(logit_yx)
                    
                    # Permute rows of Bx (node permutation)
                    bx_perm = bx_batch.clone()
                    for j in range(bx_perm.shape[0]):
                        row_perm = torch.randperm(bx_perm.shape[1])
                        bx_perm[j] = bx_perm[j, row_perm, :]
                    
                    l_perm, aux_perm = model.forward_direction(bx_perm, by_batch, return_aux=True)
                    logit_perm = model(bx_perm, by_batch)
                    prob_perm = torch.sigmoid(logit_perm)
                    
                # 1. C_yx ≈ C_xy^T
                C_xy = aux1['C_base']
                C_yx = aux2['C_base']
                C_err = torch.abs(C_yx - C_xy.transpose(1, 2)).view(bx_batch.shape[0], -1).max(dim=1)[0].numpy()
                
                # 2. P_yx ≈ P_xy^T
                P_xy = aux1['P']
                P_yx = aux2['P']
                P_err = torch.abs(P_yx - P_xy.transpose(1, 2)).view(bx_batch.shape[0], -1).max(dim=1)[0].numpy()
                
                # 3. Matching Cost Invariant
                mcost_xy = torch.sum(P_xy * C_xy, dim=(1,2)).numpy()
                mcost_yx = torch.sum(P_yx * C_yx, dim=(1,2)).numpy()
                mcost_err = np.abs(mcost_xy - mcost_yx)
                
                # 4. Equality Score Invariant
                eq_xy = torch.sum(P_xy * torch.exp(-model.beta * C_xy), dim=(1,2)).numpy()
                eq_yx = torch.sum(P_yx * torch.exp(-model.beta * C_yx), dim=(1,2)).numpy()
                eq_err = np.abs(eq_xy - eq_yx)
                
                # 5 & 6. Logits & Probs Invariant
                logit_swap_err = torch.abs(logit_xy - logit_yx).numpy()
                prob_swap_err = torch.abs(prob_xy - prob_yx).numpy()
                logit_perm_err = torch.abs(logit_xy - logit_perm).numpy()
                prob_perm_err = torch.abs(prob_xy - prob_perm).numpy()
                
                # 8, 9, 10. Sinkhorn Residuals
                mask_x = aux1['mask_x']
                mask_y = aux1['mask_y']
                
                # Row marginal sum should equal mask_x
                row_sum = P_xy.sum(dim=2)
                row_res = torch.abs(row_sum - mask_x).max(dim=1)[0].numpy()
                
                # Total mass should equal mask_x sum
                total_mass = P_xy.sum(dim=(1,2))
                exp_mass = mask_x.sum(dim=1)
                mass_res = torch.abs(total_mass - exp_mass).numpy()
                
                for j in range(bx_batch.shape[0]):
                    errs = {
                        "C_err": C_err[j],
                        "P_err": P_err[j],
                        "mcost_err": mcost_err[j],
                        "eq_err": eq_err[j],
                        "logit_swap_err": logit_swap_err[j],
                        "prob_swap_err": prob_swap_err[j],
                        "logit_perm_err": logit_perm_err[j],
                        "prob_perm_err": prob_perm_err[j],
                        "row_res": row_res[j],
                        "mass_res": mass_res[j]
                    }
                    
                    max_err_val = -1
                    failed_q = None
                    for k, v in errs.items():
                        if v > max_err_val:
                            max_err_val = v
                            failed_q = k
                            
                    all_errors.append(float(max_err_val))
                    
                    if max_err_val > worst_err:
                        worst_err = max_err_val
                        results['worst_sample_id'] = splits[split_name][i+j].get('sample_id', str(i+j))
                        results['worst_split'] = split_name
                        results['worst_seed'] = seed
                        results['failed_quantity'] = failed_q
                        
                    if max_err_val > 1e-6:
                        results['num_failures'] += 1
                        
                    results['num_audited'] += 1
                
    if len(all_errors) > 0:
        results['mean_error'] = float(np.mean(all_errors))
        results['max_error'] = float(np.max(all_errors))
        results['p95_error'] = float(np.percentile(all_errors, 95))
        results['p99_error'] = float(np.percentile(all_errors, 99))
        
    os.makedirs("results/phase3", exist_ok=True)
    with open("results/phase3/numerical_audit.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Numerical Audit Results:")
    print(json.dumps(results, indent=4))
    
    if results['num_failures'] > 0:
        print("FAIL: Numerical audit failed on some samples!")
    else:
        print("PASS: Numerical audit succeeded.")

if __name__ == "__main__":
    audit_numerics()
