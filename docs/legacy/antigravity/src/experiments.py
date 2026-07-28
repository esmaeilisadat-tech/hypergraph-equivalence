import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from dataset import generate_for_class, HGDataset
from curvature import compute_orc
from models import MLPBaseline, IncidenceHGNN

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

def test_mathematical_oracle(num_nodes=20):
    print("Validating Curvature Properties...")
    set_seed(42)
    for c in range(4):
        _, edges, x, y, actual_c = generate_for_class(c, num_nodes=num_nodes, ood=False)
        assert actual_c == c
        en_orc = compute_orc(edges, num_nodes, x, y, measure_type='EN')
        ee_orc = compute_orc(edges, num_nodes, x, y, measure_type='EE')
        
        if c == 0:
            assert en_orc < 1.0 - 1e-8
        elif c == 1:
            assert np.isclose(en_orc, 1.0, atol=1e-8)
            assert ee_orc < 1.0 - 1e-8
        elif c == 2 or c == 3:
            assert np.isclose(en_orc, 1.0, atol=1e-8)
            assert np.isclose(ee_orc, 1.0, atol=1e-8)
            
    print("Curvature assertions passed!")

def run_ml_baseline(train_dataset, test_dataset):
    X_train = [d['handcrafted'].numpy() for d in train_dataset]
    y_train = [d['label'].numpy() for d in train_dataset]
    X_test = [d['handcrafted'].numpy() for d in test_dataset]
    y_test = [d['label'].numpy() for d in test_dataset]
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    return acc, f1

def train_nn(model, train_loader, test_loader, device, epochs=30):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            H = batch['H'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            
            if isinstance(model, MLPBaseline):
                outputs = model(batch['A_feat'].to(device))
            else:
                outputs = model(H, batch['X'].to(device))
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            H = batch['H'].to(device)
            if isinstance(model, MLPBaseline):
                outputs = model(batch['A_feat'].to(device))
            else:
                outputs = model(H, batch['X'].to(device))
                
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['label'].numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    # Focus on Class 2 vs 3
    # Just return the confusion matrix so we can compute it later
    return acc, f1, confusion_matrix(all_labels, all_preds, labels=[0,1,2,3])

def run_experiments():
    test_mathematical_oracle()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    seeds = [42, 100, 2026, 777, 999]
    results = {'RF': {'acc': [], 'f1': []}, 
               'MLP': {'acc': [], 'f1': [], 'c23_f1': []}, 
               'HGNN': {'acc': [], 'f1': [], 'c23_f1': []}}
    
    for i, seed in enumerate(seeds):
        print(f"\\n--- Running Seed {seed} ---")
        set_seed(seed)
        
        # Train dataset (IID templates)
        train_dataset = HGDataset(n=1000, ood=False)
        # Test dataset (OOD templates)
        test_dataset = HGDataset(n=500, ood=True)
        
        # Random Forest Baseline
        rf_acc, rf_f1 = run_ml_baseline(train_dataset, test_dataset)
        results['RF']['acc'].append(rf_acc)
        results['RF']['f1'].append(rf_f1)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # MLP
        mlp = MLPBaseline().to(device)
        acc, f1, cm = train_nn(mlp, train_loader, test_loader, device, epochs=30)
        results['MLP']['acc'].append(acc)
        results['MLP']['f1'].append(f1)
        
        # Class 2/3 F1
        # cm is 4x4. True is row, Pred is col
        # We can calculate F1 for class 2 and 3 specifically from the confusion matrix
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # F1 score for class 2
            tp2 = cm[2,2]; fp2 = cm[:,2].sum() - tp2; fn2 = cm[2,:].sum() - tp2
            f1_2 = 2*tp2 / (2*tp2 + fp2 + fn2 + 1e-8)
            # F1 score for class 3
            tp3 = cm[3,3]; fp3 = cm[:,3].sum() - tp3; fn3 = cm[3,:].sum() - tp3
            f1_3 = 2*tp3 / (2*tp3 + fp3 + fn3 + 1e-8)
        results['MLP']['c23_f1'].append((f1_2 + f1_3)/2)
        
        # HGNN
        hgnn = IncidenceHGNN().to(device)
        acc, f1, cm = train_nn(hgnn, train_loader, test_loader, device, epochs=30)
        results['HGNN']['acc'].append(acc)
        results['HGNN']['f1'].append(f1)
        
        tp2 = cm[2,2]; fp2 = cm[:,2].sum() - tp2; fn2 = cm[2,:].sum() - tp2
        f1_2 = 2*tp2 / (2*tp2 + fp2 + fn2 + 1e-8)
        tp3 = cm[3,3]; fp3 = cm[:,3].sum() - tp3; fn3 = cm[3,:].sum() - tp3
        f1_3 = 2*tp3 / (2*tp3 + fp3 + fn3 + 1e-8)
        results['HGNN']['c23_f1'].append((f1_2 + f1_3)/2)
        
        if i == 0:
            print("HGNN Confusion Matrix (Seed 42):\\n", cm)
            
    print("\\n=== Final Results (OOD Test Set, 5 Seeds) ===")
    for model in ['RF', 'MLP', 'HGNN']:
        acc_mean = np.mean(results[model]['acc']) * 100
        acc_std = np.std(results[model]['acc']) * 100
        f1_mean = np.mean(results[model]['f1']) * 100
        f1_std = np.std(results[model]['f1']) * 100
        print(f"{model} Accuracy: {acc_mean:.1f}% ± {acc_std:.1f}%")
        print(f"{model} Macro F1: {f1_mean:.1f}% ± {f1_std:.1f}%")
        if model != 'RF':
            c23_mean = np.mean(results[model]['c23_f1']) * 100
            c23_std = np.std(results[model]['c23_f1']) * 100
            print(f"{model} Class 2&3 F1: {c23_mean:.1f}% ± {c23_std:.1f}%")

if __name__ == '__main__':
    run_experiments()
