import random
import numpy as np
from collections import Counter
from fractions import Fraction

# Import mathematical definitions from curvature.py
from curvature import get_neighbors, ee_measure, en_measure

def get_incident_neighbor_sets(H_edges, x):
    res = []
    for e in H_edges:
        if x in e:
            s = set(e)
            s.remove(x)
            res.append(frozenset(s))
    return sorted(res, key=lambda s: (len(s), sorted(list(s))))

def strongly_structurally_equivalent(H_edges, x, y):
    sx = get_incident_neighbor_sets(H_edges, x)
    sy = get_incident_neighbor_sets(H_edges, y)
    return Counter(sx) == Counter(sy)

def classify_pair(H_edges, x, y):
    nx = get_neighbors(H_edges, x)
    ny = get_neighbors(H_edges, y)
    if nx != ny:
        return 0
    mu_x = ee_measure(H_edges, x)
    mu_y = ee_measure(H_edges, y)
    if mu_x != mu_y:
        return 1
    if not strongly_structurally_equivalent(H_edges, x, y):
        return 2
    return 3

def _generate_hypergraph(c, num_nodes=20, ood=False):
    edges = []
    x, y = random.sample(range(num_nodes), 2)
    
    # Available nodes for neighbors
    pool = [n for n in range(num_nodes) if n != x and n != y]
    
    if c == 0:
        # N(x) != N(y)
        num_n = random.randint(2, 4)
        nx = random.sample(pool, num_n)
        ny = random.sample(pool, num_n)
        if set(nx) == set(ny):
            ny[0] = pool[-(num_n+1)] # Break equality
        edges.append([x] + nx)
        edges.append([y] + ny)
        
    elif c == 1:
        num_n = random.randint(3, 5)
        n_all = random.sample(pool, num_n)
        if not ood:
            # Train templates
            if random.random() < 0.5:
                edges.append([x] + n_all)
                edges.append([y] + n_all[:2])
                edges.append([y] + n_all[2:])
            else:
                edges.append([x] + n_all[:2])
                edges.append([x] + n_all[2:])
                edges.append([y, n_all[0]])
                edges.append([y] + n_all[1:])
        else:
            # OOD templates
            # e.g., overlapping triplets for x, simple partition for y
            edges.append([x, n_all[0], n_all[1], n_all[2]])
            edges.append([x, n_all[1], n_all[2], n_all[-1]])
            edges.append([y] + n_all[:2])
            edges.append([y] + n_all[2:])
            
    elif c == 2:
        num_n = random.randint(3, 4)
        n_all = random.sample(pool, num_n)
        if not ood:
            # Train templates: 1 large vs multiple pairwise
            if random.random() < 0.5:
                edges.append([x] + n_all)
                for z in n_all:
                    edges.append([y, z])
            else:
                n_all = random.sample(pool, 3)
                a, b, c_n = n_all
                edges.append([x, a, b])
                edges.append([x, b, c_n])
                edges.append([x, a, c_n])
                edges.append([y, a])
                edges.append([y, b])
                edges.append([y, c_n])
        else:
            # OOD templates: dx = dy = 4, but different incidence structures
            n_all = random.sample(pool, 3)
            a, b, c_n = n_all
            # x: {a,b}, {a,b}, {c_n}, {c_n}
            edges.append([x, a, b])
            edges.append([x, a, b])
            edges.append([x, c_n])
            edges.append([x, c_n])
            
            # y: {a,c_n}, {b,c_n}, {a,b}, {c_n}
            edges.append([y, a, c_n])
            edges.append([y, b, c_n])
            edges.append([y, a, b])
            edges.append([y, c_n])
            
    elif c == 3:
        num_n = random.randint(3, 5)
        n_all = random.sample(pool, num_n)
        if not ood:
            if random.random() < 0.5:
                edges.append([x] + n_all)
                edges.append([y] + n_all)
            else:
                edges.append([x] + n_all[:2])
                edges.append([x] + n_all[2:])
                edges.append([y] + n_all[:2])
                edges.append([y] + n_all[2:])
        else:
            # OOD templates: new symmetric structures
            a, b, c_n = random.sample(pool, 3)
            # 2 copies of {x, a, b}, 1 copy of {x, c_n}
            edges.append([x, a, b])
            edges.append([x, a, b])
            edges.append([x, c_n])
            
            edges.append([y, a, b])
            edges.append([y, a, b])
            edges.append([y, c_n])

    # Add structured noise that touches x and y without breaking class constraints
    for _ in range(10):
        sz = random.randint(2, 5)
        bg = random.sample(pool, min(sz, len(pool)))
        
        noise_type = random.randint(0, 1)
        if noise_type == 0:
            # Noise not touching x or y
            edges.append(bg)
        elif noise_type == 1:
            # Symmetric noise (preserves strong equivalence)
            if len(bg) > 0:
                edges.append([x] + bg)
                edges.append([y] + bg)
        
    return edges, x, y

def generate_for_class(target_class, num_nodes=20, ood=False):
    for _ in range(200):
        edges, x, y = _generate_hypergraph(target_class, num_nodes, ood)
        actual_class = classify_pair(edges, x, y)
        if actual_class == target_class:
            max_edges = 25
            if len(edges) <= max_edges:
                H = np.zeros((num_nodes, max_edges), dtype=np.float32)
                for e_idx, e in enumerate(edges):
                    for v in e:
                        H[v, e_idx] = 1.0
                return H, edges, x, y, actual_class
    raise RuntimeError(f"Could not generate a valid sample for class {target_class}")

def extract_handcrafted_features(edges, x, y, num_nodes):
    dx = sum(1 for e in edges if x in e)
    dy = sum(1 for e in edges if y in e)
    nx = len(get_neighbors(edges, x))
    ny = len(get_neighbors(edges, y))
    
    # Shared edges
    shared = sum(1 for e in edges if x in e and y in e)
    
    # Histogram of incident hyperedge sizes
    sizes_x = [len(e) for e in edges if x in e]
    sizes_y = [len(e) for e in edges if y in e]
    
    hist_x = [sizes_x.count(s) for s in range(2, 6)]
    hist_y = [sizes_y.count(s) for s in range(2, 6)]
    
    features = [dx, dy, nx, ny, shared] + hist_x + hist_y
    return np.array(features, dtype=np.float32)

import torch
from torch.utils.data import Dataset

class HGDataset(Dataset):
    def __init__(self, n=1500, num_nodes=20, ood=False):
        self.data = []
        counts = {0:0, 1:0, 2:0, 3:0}
        for _ in range(n):
            c = random.randint(0, 3)
            H, edges, x, y, actual = generate_for_class(c, num_nodes, ood)
            feats = extract_handcrafted_features(edges, x, y, num_nodes)
            
            # We also need to return x and y to the model
            # For HGNN we just need to build X matrix with one-hot for x and y
            X = np.zeros((num_nodes, 3), dtype=np.float32)
            X[x, 0] = 1.0
            X[y, 1] = 1.0
            X[:, 2] = 1.0
            
            # For MLP, A matrix
            H_sub = H[:, :len(edges)]
            A = H_sub @ H_sub.T
            A_feat = np.concatenate([A[x, :], A[y, :]])
            
            self.data.append({
                'H': torch.tensor(H),
                'X': torch.tensor(X),
                'A_feat': torch.tensor(A_feat),
                'handcrafted': torch.tensor(feats),
                'label': torch.tensor(actual, dtype=torch.long),
                'x': x,
                'y': y
            })
            counts[actual] += 1
        print(f"Dataset (OOD={ood}) class counts:", counts)
        
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
