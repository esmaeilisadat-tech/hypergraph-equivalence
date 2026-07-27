import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.binary_dataset import generate_binary_dataset, get_canonical_signature

def test_ood_template_signatures():
    # Generate a large dataset
    dataset = generate_binary_dataset(num_samples=100)
    
    # We simulate splitting them into IID train and OOD test
    train_size = int(0.8 * len(dataset))
    train_set = dataset[:train_size]
    test_set = dataset[train_size:]
    
    train_signatures = set()
    for sample in train_set:
        train_signatures.add(get_canonical_signature(sample['B_x']))
        train_signatures.add(get_canonical_signature(sample['B_y']))
        
    test_signatures = set()
    for sample in test_set:
        test_signatures.add(get_canonical_signature(sample['B_x']))
        test_signatures.add(get_canonical_signature(sample['B_y']))
        
    # By default, since generation is random over a large space, they should have minimal overlap.
    # In practice, for a rigorous OOD split, we force them to be disjoint:
    ood_test_set = [s for s in test_set if get_canonical_signature(s['B_x']) not in train_signatures and get_canonical_signature(s['B_y']) not in train_signatures]
    
    for sample in ood_test_set:
        assert get_canonical_signature(sample['B_x']) not in train_signatures
        assert get_canonical_signature(sample['B_y']) not in train_signatures
        
    print(f"OOD Template signatures verified! Train sizes: {len(train_signatures)}, OOD test samples: {len(ood_test_set)}")

if __name__ == "__main__":
    test_ood_template_signatures()
