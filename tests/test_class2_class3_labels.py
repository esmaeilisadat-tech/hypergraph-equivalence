import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.curvature import get_neighbors, ee_measure, en_measure, compute_orc
from src.binary_dataset import extract_bipartite_incidence, strong_structural_equivalence_multiset

def test_class3_definition():
    """
    Test that strong structural equivalence definition matches the class 3 definition:
    - Identical EN measures
    - Identical EE measures
    - Equal multiset of row supports
    """
    x, y = 0, 1
    # Example 1: True Class 3
    # Same hyperedges around x and y
    H_edges_class3 = [
        [0, 2, 3], [0, 4, 5],
        [1, 2, 3], [1, 4, 5],
        [2, 3, 4], [3, 4, 5] # background noise
    ]
    
    Bx, By, U = extract_bipartite_incidence(H_edges_class3, x, y)
    
    # Assert Class 3 properties
    assert set(get_neighbors(H_edges_class3, x)) == set(get_neighbors(H_edges_class3, y))
    
    mu_x_en = en_measure(H_edges_class3, x)
    mu_y_en = en_measure(H_edges_class3, y)
    assert mu_x_en == mu_y_en
    
    mu_x_ee = ee_measure(H_edges_class3, x)
    mu_y_ee = ee_measure(H_edges_class3, y)
    assert mu_x_ee == mu_y_ee
    
    assert strong_structural_equivalence_multiset(Bx, By) == True

def test_class2_definition():
    """
    Test that Class 2 satisfies EN/EE equivalent but NOT strongly equivalent.
    """
    x, y = 0, 1
    # We need a template where EE is identical but S(x) != S(y).
    # Since we need the *exact* same histograms, degrees, and identical universe U.
    # U = {2, 3, 4, 5}
    # x incident to: {2, 3}, {4, 5}, {2, 4}, {3, 5}  (sizes: 2, 2, 2, 2)
    # y incident to: {2, 3, 4}, {5}, {2}, {3, 4, 5} (sizes: 3, 1, 1, 3) 
    #   Wait, histograms must match exactly!
    
    # Let's try:
    # x: {2, 3}, {3, 4}, {4, 5}, {5, 2} (cycle)
    # y: {2, 4}, {4, 2}, {3, 5}, {5, 3} (diagonals duplicated) -> this is a valid one!
    
    H_edges_class2 = [
        [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 2],
        [1, 2, 4], [1, 2, 4], [1, 3, 5], [1, 3, 5]
    ]
    
    Bx, By, U = extract_bipartite_incidence(H_edges_class2, x, y)
    
    # EN measures: x has 4 neighbors (2,3,4,5). y has 4 neighbors (2,3,4,5). Uniform 1/4.
    mu_x_en = en_measure(H_edges_class2, x)
    mu_y_en = en_measure(H_edges_class2, y)
    assert mu_x_en == mu_y_en
    
    # EE measures:
    # For x: degree is 4.
    # w_x(2) = 1/4 * (1/(2-1) + 1/(2-1)) = 1/4 * (1 + 1) = 2/4 = 1/2
    # w_x(3) = 1/4 * (1 + 1) = 1/2
    # w_y(2) = 1/4 * (1/(2-1) + 1/(2-1)) = 1/4 * (1 + 1) = 1/2
    # Wait, 1/2 + 1/2 + 1/2 + 1/2 = 2. But probability must sum to 1!
    # Ah, the EE random walk weight formulation from paper:
    # P(x -> z) = 1/dx * sum_{e \ni x, z} 1/(|e|-1)
    # Here dx = 4. 
    # For x, neighbor 2 is in 2 edges of size 3 (i.e. |e|-1 = 2).
    # So P(x -> 2) = 1/4 * (1/2 + 1/2) = 1/4 = 0.25.
    # Same for 3, 4, 5. So mu_x_ee is uniform 0.25.
    
    # For y: degree is 4.
    # y has edges {1,2,4}, {1,2,4}, {1,3,5}, {1,3,5}. (all size 3, |e|-1 = 2).
    # neighbor 2 is in 2 edges of size 3.
    # P(y -> 2) = 1/4 * (1/2 + 1/2) = 1/4 = 0.25.
    # Same for 4. Same for 3 and 5. So mu_y_ee is uniform 0.25!
    
    mu_x_ee = ee_measure(H_edges_class2, x)
    mu_y_ee = ee_measure(H_edges_class2, y)
    assert mu_x_ee == mu_y_ee
    
    # So both EN and EE measures are perfectly identical!
    # Let's check histograms.
    # x edge sizes (excluding x): 2, 2, 2, 2
    # y edge sizes (excluding y): 2, 2, 2, 2
    # Both have exactly identical histograms!
    
    # Check distinct neighbors count: both 4.
    # Check degree: both 4.
    
    # Now check strong equivalence:
    assert strong_structural_equivalence_multiset(Bx, By) == False
    print("Class 2 and Class 3 labels correctly implemented!")

if __name__ == "__main__":
    test_class3_definition()
    test_class2_definition()
