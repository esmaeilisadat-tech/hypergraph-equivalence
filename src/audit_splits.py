import json
import hashlib
import os

def deterministic_hash(data):
    s = json.dumps(data, sort_keys=True)
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def get_hashes(sample):
    bx = sample['B_x']
    by = sample['B_y']
    label = sample['label']
    matched_pair_id = sample.get('matched_pair_id', '')
    template_family = sample.get('template_family', sample.get('signature', ''))
    
    struct_hash = deterministic_hash([bx, by])
    labeled_hash = deterministic_hash([bx, by, label])
    meta_hash = deterministic_hash([bx, by, label, matched_pair_id, template_family])
    
    hash1 = deterministic_hash([bx, by])
    hash2 = deterministic_hash([by, bx])
    pair_hash = min(hash1, hash2)
    
    # We also use matched_pair_id as a stability ID if available
    sample_id = sample.get('sample_id', meta_hash)
    
    return struct_hash, labeled_hash, meta_hash, pair_hash, sample_id, matched_pair_id

def load_data():
    splits = {}
    for name in ['train', 'validation', 'iid_test', 'ood_template_test', 'ood_size_test']:
        filepath = f"data/splits/{name}.json"
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                splits[name] = json.loads(f.read())['data']
        else:
            print(f"Warning: {filepath} not found.")
    return splits

def main():
    splits = load_data()
    split_names = list(splits.keys())
    
    audit_results = {
        "split_counts": {},
        "unique_sample_ids": {},
        "overlap_counts": {},
        "duplicates_within_split": {},
        "label_distributions": {}
    }
    
    all_sample_ids = {name: set() for name in split_names}
    all_struct_hashes = {name: set() for name in split_names}
    all_pair_hashes = {name: set() for name in split_names}
    all_matched_ids = {name: set() for name in split_names}
    
    for name, data in splits.items():
        audit_results["split_counts"][name] = len(data)
        
        sample_ids = []
        labels = []
        for s in data:
            struct_hash, labeled_hash, meta_hash, pair_hash, sample_id, matched_pair_id = get_hashes(s)
            sample_ids.append(sample_id)
            all_sample_ids[name].add(sample_id)
            all_struct_hashes[name].add(struct_hash)
            all_pair_hashes[name].add(pair_hash)
            if matched_pair_id:
                all_matched_ids[name].add(matched_pair_id)
            labels.append(s['label'])
            
        audit_results["unique_sample_ids"][name] = len(all_sample_ids[name])
        audit_results["duplicates_within_split"][name] = len(data) - len(all_sample_ids[name])
        
        # Label distribution
        lbl_count = {}
        for l in labels:
            lbl_count[str(l)] = lbl_count.get(str(l), 0) + 1
        audit_results["label_distributions"][name] = lbl_count

    # Overlap checks
    has_fatal_overlap = False
    for i in range(len(split_names)):
        for j in range(i+1, len(split_names)):
            n1 = split_names[i]
            n2 = split_names[j]
            pair_name = f"{n1}_vs_{n2}"
            
            overlap_sample_id = len(all_sample_ids[n1].intersection(all_sample_ids[n2]))
            overlap_struct = len(all_struct_hashes[n1].intersection(all_struct_hashes[n2]))
            overlap_pair = len(all_pair_hashes[n1].intersection(all_pair_hashes[n2]))
            overlap_matched_id = len(all_matched_ids[n1].intersection(all_matched_ids[n2]))
            
            audit_results["overlap_counts"][pair_name] = {
                "sample_id": overlap_sample_id,
                "structure_hash": overlap_struct,
                "pair_hash": overlap_pair,
                "matched_pair_id": overlap_matched_id
            }
            
            if overlap_struct > 0 or overlap_pair > 0 or overlap_matched_id > 0:
                print(f"FATAL OVERLAP detected between {n1} and {n2}: struct={overlap_struct}, pair={overlap_pair}, matched_id={overlap_matched_id}")
                has_fatal_overlap = True

    os.makedirs("results/phase3", exist_ok=True)
    with open("results/phase3/split_overlap_audit.json", "w") as f:
        json.dump(audit_results, f, indent=4)
        
    print("Audit written to results/phase3/split_overlap_audit.json")
    if has_fatal_overlap:
        raise ValueError("Strict disjointness failed. Splits contain overlapping structures or pair IDs.")
    else:
        print("All splits are disjoint. Integrity check PASS.")

if __name__ == "__main__":
    main()
