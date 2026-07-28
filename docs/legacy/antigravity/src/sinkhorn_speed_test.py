import os
import time
import json
import torch
import numpy as np

import sys
sys.path.append(os.path.abspath('.'))
from src.run_phase3_5seed import load_data, get_batch
from src.differentiable_matching import DifferentiableIncidenceMatchingNetwork

def reference_sinkhorn(C, mask_x, mask_y, eps=1e-2, num_iters=50):
    B, max_m, _ = C.shape
    P_out = torch.zeros_like(C)
    for b in range(B):
        mx = mask_x[b].sum().int().item()
        my = mask_y[b].sum().int().item()
        if mx == 0 or my == 0: continue
        
        C_b = C[b, :mx, :my]
        log_K = -C_b / eps
        log_a = -torch.log(torch.tensor(float(mx), dtype=C.dtype, device=C.device))
        log_b = -torch.log(torch.tensor(float(my), dtype=C.dtype, device=C.device))
        
        u = torch.zeros(mx, dtype=C.dtype, device=C.device)
        v = torch.zeros(my, dtype=C.dtype, device=C.device)
        
        for _ in range(num_iters):
            u = log_a - torch.logsumexp(log_K + v.unsqueeze(0), dim=1)
            v = log_b - torch.logsumexp(log_K + u.unsqueeze(1), dim=0)
            
        P_b = torch.exp(log_K + u.unsqueeze(1) + v.unsqueeze(0))
        P_out[b, :mx, :my] = P_b
    return P_out

def run_speed_test():
    splits = load_data()
    val_data = splits['validation']
    Bx, By, y = get_batch(val_data)
    
    # Init model
    model = DifferentiableIncidenceMatchingNetwork(hidden_dim=32, temp_start=0.05, use_learned_cost=False)
    model.eval()
    
    # We will test on a batch of 64
    Bx = Bx[:64]
    By = By[:64]
    
    with torch.no_grad():
        Bx_exp = Bx.unsqueeze(2)
        By_exp = By.unsqueeze(1)
        C = torch.sum(torch.abs(Bx_exp - By_exp), dim=-1)
        mask_x = (Bx.sum(dim=-1) > 0).float()
        mask_y = (By.sum(dim=-1) > 0).float()
        
        # Test Numerical Equivalence
        P_vec = model.sinkhorn(C, mask_x, mask_y, eps=model.temp, num_iters=50)
        P_ref = reference_sinkhorn(C, mask_x, mask_y, eps=model.temp, num_iters=50)
        
        is_close = torch.allclose(P_vec, P_ref, atol=1e-6, rtol=1e-4)
        
        # Warmup
        for _ in range(10):
            model.sinkhorn(C, mask_x, mask_y, eps=model.temp, num_iters=50)
            reference_sinkhorn(C, mask_x, mask_y, eps=model.temp, num_iters=50)
            
        # Timings
        reps = 30
        vec_times = []
        ref_times = []
        
        for _ in range(reps):
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.sinkhorn(C, mask_x, mask_y, eps=model.temp, num_iters=50)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            vec_times.append(time.perf_counter() - t0)
            
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.perf_counter()
            reference_sinkhorn(C, mask_x, mask_y, eps=model.temp, num_iters=50)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            ref_times.append(time.perf_counter() - t0)
            
    vec_med = float(np.median(vec_times))
    ref_med = float(np.median(ref_times))
    
    results = {
        'numerical_equivalence_status': 'PASS' if is_close else 'FAIL',
        'max_diff': float(torch.abs(P_vec - P_ref).max()),
        'vectorized_runtime_median': vec_med,
        'vectorized_runtime_iqr': float(np.percentile(vec_times, 75) - np.percentile(vec_times, 25)),
        'reference_runtime_median': ref_med,
        'reference_runtime_iqr': float(np.percentile(ref_times, 75) - np.percentile(ref_times, 25)),
        'speedup_ratio': ref_med / vec_med if vec_med > 0 else 0
    }
    
    with open('results/phase3/sinkhorn_equivalence_speed.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Numerical Equivalence: {results['numerical_equivalence_status']} (Max Diff: {results['max_diff']:.2e})")
    print(f"Speedup: {results['speedup_ratio']:.2f}x (Vec: {vec_med*1000:.2f}ms, Ref: {ref_med*1000:.2f}ms)")

if __name__ == '__main__':
    run_speed_test()
