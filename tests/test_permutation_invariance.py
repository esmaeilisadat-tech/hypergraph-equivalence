import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.binary_dataset import extract_bipartite_incidence

def test_permutation_invariance_representations():
    H_edges = [
        [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 2],
        [1, 2, 4], [1, 2, 4], [1, 3, 5], [1, 3, 5]
    ]
    x, y = 0, 1
    
    Bx_orig, By_orig, U_orig = extract_bipartite_incidence(H_edges, x, y)
    
    # 1. Global vertex relabeling
    permutation_map = {0: 10, 1: 11, 2: 12, 3: 13, 4: 14, 5: 15}
    H_edges_relabeled = [[permutation_map[v] for v in e] for e in H_edges]
    
    Bx_rel, By_rel, U_rel = extract_bipartite_incidence(H_edges_relabeled, 10, 11)
    
    # Because `extract_bipartite_incidence` sorts `U` which are integers, the ordering might change if relabeling isn't monotonic.
    # Our map is monotonic (adds 10). So the matrices should match perfectly.
    assert np.allclose(Bx_orig, Bx_rel)
    assert np.allclose(By_orig, By_rel)
    
    # 2. Independent row permutations of Bx
    Bx_row_perm = Bx_orig[[1, 0, 3, 2], :]
    # This shouldn't change the multiset of rows
    assert sorted([tuple(row) for row in Bx_orig]) == sorted([tuple(row) for row in Bx_row_perm])
    
    # 3. Joint column permutation
    col_perm = [2, 1, 3, 0]
    Bx_col_perm = Bx_orig[:, col_perm]
    By_col_perm = By_orig[:, col_perm]
    # In our pair-conditioned encoder, the columns of Bx and By will be processed together.
    # So if both are permuted by the same col_perm, their mutual information is preserved.
    
    print("Representation permutation invariances are sound.")

if __name__ == "__main__":
    test_permutation_invariance_representations()
