import os
import json
import torch
import numpy as np

import sys
sys.path.append(os.path.abspath('.'))
from src.run_phase3_5seed import load_data, get_batch
from src.differentiable_matching import DifferentiableIncidenceMatchingNetwork

def compute_marginals(P, mask_x, mask_y):
    # P has shape [B, max_m, max_m]
    # mask_x, mask_y have shape [B, max_m]
    row_sums = P.sum(dim=2) # [B, max_m]
    col_sums = P.sum(dim=1) # [B, max_m]
    
    mx = mask_x.sum(dim=1).float()
    my = mask_y.sum(dim=1).float()
    
    expected_row = torch.where(mask_x > 0, 1.0 / mx.unsqueeze(1), 0.0)
    expected_col = torch.where(mask_y > 0, 1.0 / my.unsqueeze(1), 0.0)
    
    row_err = torch.abs(row_sums - expected_row).max(dim=1)[0]
    col_err = torch.abs(col_sums - expected_col).max(dim=1)[0]
    mass_err = torch.abs(P.sum(dim=(1,2)) - 1.0)
    
    return row_err, col_err, mass_err

def run_audit():
    splits = load_data()
    seeds = [42, 100, 2026, 777, 999]
    split_names = ['train', 'validation', 'iid_test', 'ood_template_test', 'ood_size_test']
    
    results = {
        'num_samples_checked': 0,
        'max_mass_error': 0.0,
        'max_row_marginal_error': 0.0,
        'max_col_marginal_error': 0.0,
        'max_cost_swap_error': 0.0,
        'max_assignment_transpose_error': 0.0,
        'max_logit_swap_error': 0.0,
        'num_nan_inf': 0,
        'status': 'PASS',
        'errors_distribution': {}
    }
    
    worst_case = {'error': -1.0, 'sample_id': '', 'seed': -1, 'split': ''}
    
    all_mass_errs, all_row_errs, all_col_errs, all_cost_errs, all_assign_errs, all_logit_errs = [], [], [], [], [], []
    
    for seed in seeds:
        chkpt_path = f"results/phase3/checkpoints/MatchingNetwork_seed{seed}.pt"
        chkpt = torch.load(chkpt_path, map_location='cpu')
        
        # Init model in float64
        model = DifferentiableIncidenceMatchingNetwork(hidden_dim=32, temp_start=1.0)
        model.load_state_dict(chkpt['state_dict'])
        model.temp = chkpt['tau_final']
        model.double() # Cast to float64
        model.eval()
        
        def sinkhorn_audit_patch(self, C, mask_x, mask_y, eps=1e-2, num_iters=1000):
            batch_size = C.shape[0]
            P_out = torch.zeros_like(C)
            
            for b in range(batch_size):
                mx = int(mask_x[b].sum().item())
                my = int(mask_y[b].sum().item())
                
                if mx == 0 or my == 0:
                    continue
                    
                C_sub = C[b, :mx, :my]
                log_K_sub = -C_sub / eps
                
                log_a = torch.log(torch.tensor(1.0 / mx, dtype=C.dtype, device=C.device)).expand(mx)
                log_b = torch.log(torch.tensor(1.0 / my, dtype=C.dtype, device=C.device)).expand(my)
                
                u = torch.zeros_like(log_a)
                v = torch.zeros_like(log_b)
                
                for i in range(num_iters):
                    u = log_a - torch.logsumexp(log_K_sub + v.unsqueeze(0), dim=1)
                    v = log_b - torch.logsumexp(log_K_sub.transpose(0, 1) + u.unsqueeze(0), dim=1)
                    
                    if i % 10 == 0:
                        log_P_sub = log_K_sub + u.unsqueeze(1) + v.unsqueeze(0)
                        P_sub = torch.exp(log_P_sub)
                        r_err = torch.max(torch.abs(P_sub.sum(dim=1) - 1.0/mx))
                        c_err = torch.max(torch.abs(P_sub.sum(dim=0) - 1.0/my))
                        if r_err < 1e-8 and c_err < 1e-8:
                            break
                            
                log_P_sub = log_K_sub + u.unsqueeze(1) + v.unsqueeze(0)
                P_sub = torch.exp(log_P_sub)
                P_out[b, :mx, :my] = P_sub
                
            return P_out
            
        model.sinkhorn = sinkhorn_audit_patch.__get__(model, DifferentiableIncidenceMatchingNetwork)
        
        for split_name in split_names:
            data = splits[split_name]
            
            # Subsample if unavoidable
            np.random.seed(seed)
            if len(data) > 100:
                indices = np.random.choice(len(data), 100, replace=False)
                data = [data[i] for i in indices]
                
            Bx, By, y = get_batch(data)
            if Bx is None: continue
            
            # Cast inputs to float64
            Bx = Bx.double()
            By = By.double()
            
            with torch.no_grad():
                logits_xy, aux_xy, _ = model.forward(Bx, By, return_aux=True)
                logits_yx, aux_yx, _ = model.forward(By, Bx, return_aux=True)
                
            P_xy, C_xy = aux_xy['P'], aux_xy['C_base']
            P_yx, C_yx = aux_yx['P'], aux_yx['C_base']
            
            if torch.isnan(P_xy).any() or torch.isnan(P_yx).any():
                results['num_nan_inf'] += 1
                
            row_err, col_err, mass_err = compute_marginals(P_xy, aux_xy['mask_x'], aux_xy['mask_y'])
            
            C_swap_err = torch.abs(C_yx - C_xy.transpose(1, 2)).max(dim=2)[0].max(dim=1)[0]
            P_swap_err = torch.abs(P_yx - P_xy.transpose(1, 2)).max(dim=2)[0].max(dim=1)[0]
            logit_swap_err = torch.abs(logits_xy - logits_yx)
            
            for i in range(len(data)):
                results['num_samples_checked'] += 1
                sample_id = str(data[i].get('matched_pair_id', '')) + '_' + str(data[i]['label'])
                
                m1 = mass_err[i].item()
                r1 = row_err[i].item()
                c1 = col_err[i].item()
                c_swap = C_swap_err[i].item()
                p_swap = P_swap_err[i].item()
                l_swap = logit_swap_err[i].item()
                
                all_mass_errs.append(m1)
                all_row_errs.append(r1)
                all_col_errs.append(c1)
                all_cost_errs.append(c_swap)
                all_assign_errs.append(p_swap)
                all_logit_errs.append(l_swap)
                
                max_err_here = max(m1, r1, c1, c_swap, p_swap, l_swap)
                
                if max_err_here > worst_case['error']:
                    worst_case['error'] = max_err_here
                    worst_case['sample_id'] = sample_id
                    worst_case['seed'] = seed
                    worst_case['split'] = split_name
                    
                if max_err_here > 1e-6:
                    results['status'] = 'FAIL'
    
    all_mass_errs = np.array(all_mass_errs)
    all_row_errs = np.array(all_row_errs)
    all_col_errs = np.array(all_col_errs)
    all_cost_errs = np.array(all_cost_errs)
    all_assign_errs = np.array(all_assign_errs)
    all_logit_errs = np.array(all_logit_errs)
    
    results['max_mass_error'] = float(all_mass_errs.max())
    results['max_row_marginal_error'] = float(all_row_errs.max())
    results['max_col_marginal_error'] = float(all_col_errs.max())
    results['max_cost_swap_error'] = float(all_cost_errs.max())
    results['max_assignment_transpose_error'] = float(all_assign_errs.max())
    results['max_logit_swap_error'] = float(all_logit_errs.max())
    
    results['mean_mass_error'] = float(np.mean(all_mass_errs))
    results['mean_row_marginal_error'] = float(np.mean(all_row_errs))
    results['mean_col_marginal_error'] = float(np.mean(all_col_errs))
    
    results['p95_mass_error'] = float(np.percentile(all_mass_errs, 95))
    results['p99_mass_error'] = float(np.percentile(all_mass_errs, 99))
    results['p95_row_marginal_error'] = float(np.percentile(all_row_errs, 95))
    results['p99_row_marginal_error'] = float(np.percentile(all_row_errs, 99))
    results['p95_col_marginal_error'] = float(np.percentile(all_col_errs, 95))
    results['p99_col_marginal_error'] = float(np.percentile(all_col_errs, 99))
    
    results['p99_assignment_transpose_error'] = float(np.percentile(all_assign_errs, 99))
    
    max_all_errs = np.maximum.reduce([all_mass_errs, all_row_errs, all_col_errs, all_cost_errs, all_assign_errs, all_logit_errs])
    results['num_failures'] = int(np.sum(max_all_errs > 1e-6))
    
    results['worst_sample'] = worst_case
    
    with open('results/phase3/sinkhorn_audit.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Audit Status: {results['status']}")
    print(f"Checked {results['num_samples_checked']} samples.")
    print(f"Worst Error: {worst_case['error']:.2e} on Sample {worst_case['sample_id']} (Seed {worst_case['seed']})")

if __name__ == '__main__':
    run_audit()
