import numpy as np
from collections import Counter
from scipy.optimize import linear_sum_assignment

def exact_counter_oracle(Bx, By):
    """
    Class 3 iff:
    Counter(frozenset(row support) for row in B_x) == Counter(frozenset(row support) for row in B_y)
    """
    def to_multiset(B):
        return Counter([frozenset(np.where(row == 1)[0]) for row in B])
        
    return 1.0 if to_multiset(Bx) == to_multiset(By) else 0.0

def hungarian_matching_oracle(Bx, By):
    """
    Class 3 iff minimum_assignment_cost == 0 using L1 distance on joint neighbor columns.
    """
    if Bx.shape != By.shape:
        return 0.0 # Cannot perfectly match if shapes differ
        
    m = Bx.shape[0]
    C = np.zeros((m, m))
    
    for i in range(m):
        for j in range(m):
            C[i, j] = np.sum(np.abs(Bx[i] - By[j]))
            
    row_ind, col_ind = linear_sum_assignment(C)
    min_cost = C[row_ind, col_ind].sum()
    
    return 1.0 if min_cost == 0 else 0.0
