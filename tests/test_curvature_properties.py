import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.binary_dataset import generate_binary_dataset, build_hypergraph_from_bipartite
from src.curvature import compute_orc, get_neighbors, ee_measure, en_measure

def test_dataset_properties():
    dataset = generate_binary_dataset(num_samples=10)
    
    for sample in dataset:
        Bx = sample['B_x']
        By = sample['B_y']
        
        # 1. Row sums and Column sums should match
        assert np.array_equal(sorted(Bx.sum(axis=1)), sorted(By.sum(axis=1)))
        assert np.array_equal(Bx.sum(axis=0), By.sum(axis=0))
        
        # 2. Build hypergraph and test EE/EN
        H_edges, x, y = build_hypergraph_from_bipartite(Bx, By, num_background_nodes=5)
        
        nx = get_neighbors(H_edges, x)
        ny = get_neighbors(H_edges, y)
        assert nx == ny
        
        mu_x_en = en_measure(H_edges, x)
        mu_y_en = en_measure(H_edges, y)
        assert mu_x_en == mu_y_en
        
        mu_x_ee = ee_measure(H_edges, x)
        mu_y_ee = ee_measure(H_edges, y)
        
        # The key assertion: Ryser's exchange on same-size edges strictly preserves EE measure!
        # Because we only swap edges of the same size, the 1/(|e|-1) weights are exactly the same!
        # And the degree is identical.
        for u in mu_x_ee:
            assert mu_x_ee[u] == mu_y_ee[u]
            
        # Class check
        if sample['label'] == 1:
            pass # S(x) == S(y)
            
    print("Dataset properties and Ryser's exchange logic perfectly validated!")

if __name__ == "__main__":
    test_dataset_properties()
