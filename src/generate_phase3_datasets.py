import os
import json
import random
import numpy as np
from uuid import uuid4
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.binary_dataset import get_canonical_signature, strong_structural_equivalence_multiset

def apply_multi_switch_ryser(B, num_switches):
    from collections import defaultdict
    B_new = B.copy()
    for _ in range(num_switches):
        row_sums = B_new.sum(axis=1)
        groups = defaultdict(list)
        for i, s in enumerate(row_sums):
            groups[s].append(i)
            
        valid_groups = [s for s, rows in groups.items() if len(rows) >= 2]
        if not valid_groups:
            continue
            
        s = random.choice(valid_groups)
        r1, r2 = random.sample(groups[s], 2)
        
        c1_candidates = np.where((B_new[r1] == 1) & (B_new[r2] == 0))[0]
        c2_candidates = np.where((B_new[r1] == 0) & (B_new[r2] == 1))[0]
        
        if len(c1_candidates) > 0 and len(c2_candidates) > 0:
            c1 = random.choice(c1_candidates)
            c2 = random.choice(c2_candidates)
            
            B_new[r1, c1] = 0
            B_new[r2, c1] = 1
            B_new[r1, c2] = 1
            B_new[r2, c2] = 0
    return B_new

def generate_base_matrix(k, m, min_size=2, max_size=5):
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

def generate_pair(k, m, min_size=2, max_size=5, num_switches=1):
    while True:
        B_x = generate_base_matrix(k, m, min_size, max_size)
        B_y_class2 = apply_multi_switch_ryser(B_x, num_switches)
        if B_y_class2 is not None and not strong_structural_equivalence_multiset(B_x, B_y_class2):
            break
            
    matched_pair_id = str(uuid4())
    sig = get_canonical_signature(B_x, B_y_class2)
    
    # Class 3
    row_perm = np.random.permutation(m)
    B_y_class3 = B_x[row_perm, :]
    
    return [
        {
            'matched_pair_id': matched_pair_id,
            'B_x': B_x.tolist(),
            'B_y': B_y_class2.tolist(),
            'label': 0,
            'k': k, 'm': m,
            'signature': sig
        },
        {
            'matched_pair_id': matched_pair_id,
            'B_x': B_x.tolist(),
            'B_y': B_y_class3.tolist(),
            'label': 1,
            'k': k, 'm': m,
            'signature': get_canonical_signature(B_x, B_y_class3)
        }
    ]

def generate_splits(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    # Generate 400 base templates for Train/Val/IID Test
    train_splits = []
    val_splits = []
    iid_test = []
    
    for _ in range(400):
        k = random.randint(5, 10)
        m = random.randint(5, 12)
        base_pair = generate_pair(k, m)
        
        # Original goes to train
        train_splits.extend(base_pair)
        
        # Generate IID variation for Val (row/col permuted)
        B_x = np.array(base_pair[0]['B_x'])
        val_pair = []
        for bp in base_pair:
            rp = np.random.permutation(m)
            cp = np.random.permutation(k)
            B_x_new = np.array(bp['B_x'])[rp][:, cp]
            B_y_new = np.array(bp['B_y'])[rp][:, cp]
            
            val_pair.append({
                'matched_pair_id': str(uuid4()), # new ID for validation
                'B_x': B_x_new.tolist(),
                'B_y': B_y_new.tolist(),
                'label': bp['label'],
                'k': k, 'm': m,
                'signature': bp['signature'] # signature is technically same
            })
        val_splits.extend(val_pair)
        
        # Generate another IID variation for IID Test
        iid_pair = []
        for bp in base_pair:
            rp = np.random.permutation(m)
            cp = np.random.permutation(k)
            B_x_new = np.array(bp['B_x'])[rp][:, cp]
            B_y_new = np.array(bp['B_y'])[rp][:, cp]
            
            iid_pair.append({
                'matched_pair_id': str(uuid4()),
                'B_x': B_x_new.tolist(),
                'B_y': B_y_new.tolist(),
                'label': bp['label'],
                'k': k, 'm': m,
                'signature': bp['signature']
            })
        iid_test.extend(iid_pair)
        
    # Generate completely new templates for OOD Template Test
    ood_template = []
    for _ in range(400):
        k = random.randint(5, 10)
        m = random.randint(5, 12)
        ood_template.extend(generate_pair(k, m))
                
    # 2. OOD Size
    ood_size = []
    for _ in range(200): # k in [12..20], m in [15..30]
        k = random.randint(12, 20)
        m = random.randint(15, 30)
        ood_size.extend(generate_pair(k, m))
        
    # 3. OOD Cardinality (Row sizes > 5)
    ood_cardinality = []
    for _ in range(200):
        k = random.randint(8, 12)
        m = random.randint(8, 12)
        ood_cardinality.extend(generate_pair(k, m, min_size=6, max_size=8))
        
    # 4. OOD Composition (Multi-switch)
    ood_composition = []
    for _ in range(200):
        k = random.randint(6, 12)
        m = random.randint(6, 12)
        ood_composition.extend(generate_pair(k, m, num_switches=5))
        
    os.makedirs("data/splits", exist_ok=True)
    
    for name, data in [
        ('train', train_splits),
        ('validation', val_splits),
        ('iid_test', iid_test),
        ('ood_template_test', ood_template),
        ('ood_size_test', ood_size),
        ('ood_cardinality_test', ood_cardinality),
        ('ood_composition_test', ood_composition)
    ]:
        with open(f"data/splits/{name}.json", "w") as f:
            json.dump({
                'seed': seed,
                'data': data
            }, f)
            
        print(f"Saved {name}: {len(data)} samples")

if __name__ == "__main__":
    generate_splits()
