import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

from src.binary_dataset import generate_matched_binary_dataset, split_dataset, extract_handcrafted_features, get_canonical_signature
from src.incidence_star_model import NestedDeepSets, PairConditionedBipartiteEncoder

def audit_shortcut_features(train_set, test_set):
    rows = []
    for split_name, ds in [('train', train_set), ('test', test_set)]:
        for s in ds:
            # compute handcrafted features
            # we need to make sure they are identical for Class 2 and Class 3
            Bx, By = s['B_x'], s['B_y']
            
            row_sums_x = sorted(Bx.sum(axis=1))
            col_sums_x = sorted(Bx.sum(axis=0))
            
            rows.append({
                'matched_pair_id': s['matched_pair_id'],
                'label': s['label'],
                'split': split_name,
                'signature': get_canonical_signature(Bx, By),
                'k': s['k'],
                'm': s['m'],
                'row_sums': str(row_sums_x),
                'col_sums': str(col_sums_x),
                'total_incidence': Bx.sum()
            })
            
    df = pd.DataFrame(rows)
    df.to_csv("results/shortcut_audit.csv", index=False)
    print(f"Shortcut audit saved. Total samples: {len(df)}")
    return df

def run_baselines(train_set, test_set):
    X_train = np.array([extract_handcrafted_features(s['B_x'], s['B_y']) for s in train_set])
    y_train = np.array([s['label'] for s in train_set])
    
    X_test = np.array([extract_handcrafted_features(s['B_x'], s['B_y']) for s in test_set])
    y_test = np.array([s['label'] for s in test_set])
    
    models = {
        'Majority Baseline': None,
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    for name, clf in models.items():
        if name == 'Majority Baseline':
            # Always predict majority class from train
            majority = np.argmax(np.bincount(y_train))
            y_pred = np.full_like(y_test, majority)
        else:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {'acc': acc, 'macro_f1': f1, 'cm': cm}
        print(f"[{name}] Acc: {acc:.4f}, Macro F1: {f1:.4f}")
        
    return results

def train_nn_model(model, train_set, test_set, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCEWithLogitsLoss()
    
    # We must pad Bx and By for batching.
    # Since they are variable size, we will pad them to max_m and max_k
    max_k = max([s['k'] for s in train_set + test_set])
    max_m = max([s['m'] for s in train_set + test_set])
    
    def get_batch(dataset):
        batch_Bx = []
        batch_By = []
        labels = []
        for s in dataset:
            Bx = s['B_x']
            By = s['B_y']
            
            pad_Bx = np.pad(Bx, ((0, max_m - Bx.shape[0]), (0, max_k - Bx.shape[1])))
            pad_By = np.pad(By, ((0, max_m - By.shape[0]), (0, max_k - By.shape[1])))
            
            batch_Bx.append(pad_Bx)
            batch_By.append(pad_By)
            labels.append(float(s['label']))
            
        return torch.tensor(np.array(batch_Bx)), torch.tensor(np.array(batch_By)), torch.tensor(labels)
        
    train_Bx, train_By, train_y = get_batch(train_set)
    test_Bx, test_By, test_y = get_batch(test_set)
    
    model.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        logits = model(train_Bx, train_By)
        loss = criterion(logits, train_y)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        test_logits = model(test_Bx, test_By)
        test_preds = (torch.sigmoid(test_logits) > 0.5).float()
        acc = accuracy_score(test_y.numpy(), test_preds.numpy())
        f1 = f1_score(test_y.numpy(), test_preds.numpy(), average='macro')
        cm = confusion_matrix(test_y.numpy(), test_preds.numpy())
        
    return acc, f1, cm

def run_phase3_one_seed():
    os.makedirs("results", exist_ok=True)
    print("Generating matched dataset...")
    dataset = generate_matched_binary_dataset(num_pairs=500)
    
    print("Splitting strictly by template signature...")
    train_set, test_set = split_dataset(dataset)
    print(f"Train size: {len(train_set)}, OOD Test size: {len(test_set)}")
    
    audit_shortcut_features(train_set, test_set)
    
    print("\\n--- Baselines on Handcrafted Shortcuts ---")
    run_baselines(train_set, test_set)
    
    print("\\n--- Neural Models ---")
    
    model_A = NestedDeepSets(hidden_dim=32)
    acc_A, f1_A, cm_A = train_nn_model(model_A, train_set, test_set, epochs=100)
    print(f"[Model A: Nested DeepSets] Acc: {acc_A:.4f}, Macro F1: {f1_A:.4f}")
    print(f"CM:\\n{cm_A}")
    
    model_B = PairConditionedBipartiteEncoder(hidden_dim=32)
    acc_B, f1_B, cm_B = train_nn_model(model_B, train_set, test_set, epochs=100)
    print(f"[Model B: Pair-Cond. Incidence-Star Encoder] Acc: {acc_B:.4f}, Macro F1: {f1_B:.4f}")
    print(f"CM:\\n{cm_B}")

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    run_phase3_one_seed()
