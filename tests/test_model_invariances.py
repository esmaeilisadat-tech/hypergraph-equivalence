import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from src.incidence_star_model import NestedDeepSets, PairConditionedBipartiteEncoder

def test_model_invariances():
    # Model Setup
    model_A = NestedDeepSets(hidden_dim=16)
    model_A.eval()
    
    model_B = PairConditionedBipartiteEncoder(hidden_dim=16)
    model_B.eval()
    
    # Generate Dummy Data
    Bx = torch.tensor([
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [1, 0, 1, 1]
    ], dtype=torch.float32).unsqueeze(0)
    
    By = torch.tensor([
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ], dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        out_A_orig = model_A(Bx, By).item()
        out_B_orig = model_B(Bx, By).item()
        
    # 1. Independent row permutation of Bx
    row_perm = [2, 0, 1]
    Bx_row = Bx[:, row_perm, :]
    with torch.no_grad():
        assert np.isclose(model_A(Bx_row, By).item(), out_A_orig, atol=1e-5)
        assert np.isclose(model_B(Bx_row, By).item(), out_B_orig, atol=1e-5)
        
    # 2. Joint column permutation
    col_perm = [3, 1, 0, 2]
    Bx_col = Bx[:, :, col_perm]
    By_col = By[:, :, col_perm]
    with torch.no_grad():
        # Model A sums over columns uniformly, so it's invariant.
        assert np.isclose(model_A(Bx_col, By_col).item(), out_A_orig, atol=1e-5)
        # Model B uses bipartite msg passing which is permutation equivariant over nodes,
        # then sums over everything. So it's invariant!
        assert np.isclose(model_B(Bx_col, By_col).item(), out_B_orig, atol=1e-5)
        
    # 3. Swap x and y
    with torch.no_grad():
        # Both models use symmetric root pair [hx+hy, |hx-hy|, hx*hy]
        assert np.isclose(model_A(By, Bx).item(), out_A_orig, atol=1e-5)
        assert np.isclose(model_B(By, Bx).item(), out_B_orig, atol=1e-5)
        
    print("Neural Model Invariances perfectly validated!")

if __name__ == "__main__":
    test_model_invariances()
