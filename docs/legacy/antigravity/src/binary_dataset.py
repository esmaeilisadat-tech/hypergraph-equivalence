import numpy as np
import random
from collections import Counter
import networkx as nx
import pandas as pd
from uuid import uuid4

def _build_bipartite_graph(Bx, By):
    G = nx.Graph()
    k = Bx.shape[1]
    mx = Bx.shape[0]
    my = By.shape[0]
    
    # Add neighbor nodes
    for j in range(k):
        G.add_node(f"n_{j}", color="neighbor")
        
    # Add X edges
    for i in range(mx):
        u = f"x_{i}"
        G.add_node(u, color="edge_x")
        for j in range(k):
            if Bx[i, j] == 1:
                G.add_edge(u, f"n_{j}")
                
    # Add Y edges
    for i in range(my):
        u = f"y_{i}"
        G.add_node(u, color="edge_y")
        for j in range(k):
            if By[i, j] == 1:
                G.add_edge(u, f"n_{j}")
    return G

_GLOBAL_TEMPLATE_REGISTRY = [] # stores (WL_hash, Graph)

def get_canonical_signature(B_x, B_y):
    """
    Computes a canonical signature invariant to:
    - independent row permutations of B_x and B_y
    - joint column permutation
    - swapping B_x and B_y
    Using Weisfeiler-Lehman graph hashing as pre-filter, followed by explicit exact isomorphism.
    """
    G1 = _build_bipartite_graph(B_x, B_y)
    G2 = _build_bipartite_graph(B_y, B_x)
    
    h1 = nx.weisfeiler_lehman_graph_hash(G1, node_attr="color")
    h2 = nx.weisfeiler_lehman_graph_hash(G2, node_attr="color")
    
    # Sort hashes to ensure swap invariance at the WL level
    hash_pair = tuple(sorted([h1, h2]))
    
    # Check registry for isomorphism
    # WL hash collisions are possible but rare. We use explicit isomorphism check.
    # To be invariant to X/Y swap, we must check if G1 is isomorphic to either G_reg1 or G_reg2.
    nm = nx.algorithms.isomorphism.categorical_node_match("color", "")
    
    for idx, (reg_hash_pair, reg_G1, reg_G2) in enumerate(_GLOBAL_TEMPLATE_REGISTRY):
        if reg_hash_pair == hash_pair:
            # Hash matches, perform expensive exact check
            if nx.is_isomorphic(G1, reg_G1, node_match=nm) or nx.is_isomorphic(G1, reg_G2, node_match=nm):
                return idx
                
    # If not found, register new template
    idx = len(_GLOBAL_TEMPLATE_REGISTRY)
    _GLOBAL_TEMPLATE_REGISTRY.append((hash_pair, G1, G2))
    return idx

def extract_bipartite_incidence(H_edges, x, y):
    n_x = set()
    n_y = set()
    for e in H_edges:
        if x in e:
            n_x.update([v for v in e if v != x])
        if y in e:
            n_y.update([v for v in e if v != y])
            
    U = sorted(list(n_x.union(n_y)))
    k = len(U)
    u_to_idx = {u: idx for idx, u in enumerate(U)}
    
    E_x = [e for e in H_edges if x in e]
    E_y = [e for e in H_edges if y in e]
    
    B_x = np.zeros((len(E_x), k), dtype=np.float32)
    B_y = np.zeros((len(E_y), k), dtype=np.float32)
    
    for i, e in enumerate(E_x):
        for v in e:
            if v != x and v in u_to_idx:
                B_x[i, u_to_idx[v]] = 1.0
                
    for i, e in enumerate(E_y):
        for v in e:
            if v != y and v in u_to_idx:
                B_y[i, u_to_idx[v]] = 1.0
                
    return B_x, B_y, U

def strong_structural_equivalence_multiset(B_x, B_y):
    if B_x.shape[1] != B_y.shape[1]:
        return False
    def to_multiset(B):
        return Counter([frozenset(np.where(row == 1)[0]) for row in B])
    return to_multiset(B_x) == to_multiset(B_y)

def apply_ryser_exchange(B):
    row_sums = B.sum(axis=1)
    from collections import defaultdict
    groups = defaultdict(list)
    for i, s in enumerate(row_sums):
        groups[s].append(i)
        
    for _ in range(100):
        valid_groups = [s for s, rows in groups.items() if len(rows) >= 2]
        if not valid_groups:
            return None
        s = random.choice(valid_groups)
        r1, r2 = random.sample(groups[s], 2)
        
        c1_candidates = np.where((B[r1] == 1) & (B[r2] == 0))[0]
        c2_candidates = np.where((B[r1] == 0) & (B[r2] == 1))[0]
        
        if len(c1_candidates) > 0 and len(c2_candidates) > 0:
            c1 = random.choice(c1_candidates)
            c2 = random.choice(c2_candidates)
            
            B_new = B.copy()
            B_new[r1, c1] = 0
            B_new[r2, c1] = 1
            B_new[r1, c2] = 1
            B_new[r2, c2] = 0
            
            return B_new
    return None

def generate_base_matrix(k, m, min_size=2, max_size=4):
    B = np.zeros((m, k), dtype=np.float32)
    for i in range(m):
        size = random.randint(min_size, max_size)
        size = min(size, k)
        cols = random.sample(range(k), size)
        B[i, cols] = 1.0
    for j in range(k):
        if B[:, j].sum() == 0:
            B[random.randint(0, m-1), j] = 1.0
    return B

def generate_matched_binary_dataset(num_pairs=500):
    dataset = []
    
    for _ in range(num_pairs):
        k = random.randint(5, 12)
        m = random.randint(5, 15)
        
        while True:
            B_x = generate_base_matrix(k, m)
            B_y_class2 = apply_ryser_exchange(B_x)
            if B_y_class2 is not None and not strong_structural_equivalence_multiset(B_x, B_y_class2):
                break
                
        matched_pair_id = str(uuid4())
        
        # Class 2
        dataset.append({
            'matched_pair_id': matched_pair_id,
            'B_x': B_x,
            'B_y': B_y_class2,
            'label': 0,
            'k': k,
            'm': m
        })
        
        # Class 3
        row_perm = np.random.permutation(m)
        B_y_class3 = B_x[row_perm, :]
        dataset.append({
            'matched_pair_id': matched_pair_id,
            'B_x': B_x,
            'B_y': B_y_class3,
            'label': 1,
            'k': k,
            'm': m
        })
        
    return dataset

def split_dataset(dataset):
    # Group by matched_pair_id
    from collections import defaultdict
    groups = defaultdict(list)
    for s in dataset:
        groups[s['matched_pair_id']].append(s)
        
    pairs = list(groups.values())
    random.shuffle(pairs)
    
    train_splits = []
    iid_val = []
    iid_test = []
    ood_splits = []
    
    train_sigs = set()
    for pair in pairs:
        sig1 = get_canonical_signature(pair[0]['B_x'], pair[0]['B_y'])
        sig2 = get_canonical_signature(pair[1]['B_x'], pair[1]['B_y'])
        
        # Determine if we should put in OOD
        if len(train_splits) < len(pairs) * 0.6:
            # Add to train
            train_splits.extend(pair)
            train_sigs.add(sig1)
            train_sigs.add(sig2)
        elif len(iid_val) < len(pairs) * 0.1 and (sig1 in train_sigs or sig2 in train_sigs):
            # Same signature as train -> IID
            iid_val.extend(pair)
        elif len(iid_test) < len(pairs) * 0.1 and (sig1 in train_sigs or sig2 in train_sigs):
            # Same signature as train -> IID
            iid_test.extend(pair)
        else:
            # Not in train -> OOD
            if sig1 not in train_sigs and sig2 not in train_sigs:
                ood_splits.extend(pair)
            else:
                # Forced to be IID due to signature overlap
                iid_test.extend(pair)
                
    return train_splits, iid_val, iid_test, ood_splits

def extract_handcrafted_features(B_x, B_y):
    """
    Extracts 1D feature vector for baselines
    """
    row_sums_x = sorted(B_x.sum(axis=1))
    col_sums_x = sorted(B_x.sum(axis=0))
    # Add simple padding for max dimensions to keep vector size constant
    max_k = 12
    max_m = 15
    rx = np.pad(row_sums_x, (0, max_m - len(row_sums_x)))
    cx = np.pad(col_sums_x, (0, max_k - len(col_sums_x)))
    return np.concatenate([rx, cx, [B_x.sum(), B_x.shape[0], B_x.shape[1]]])
