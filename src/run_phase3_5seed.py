import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import hashlib
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.incidence_star_model import NestedDeepSets, PairConditionedBipartiteEncoder
from src.differentiable_matching import DifferentiableIncidenceMatchingNetwork
from src.oracles import exact_counter_oracle, hungarian_matching_oracle

def load_data():
    splits = {}
    split_hashes = {}
    for name in ['train', 'validation', 'iid_test', 'ood_template_test', 'ood_size_test', 'ood_cardinality_test', 'ood_composition_test']:
        filepath = f"data/splits/{name}.json"
        with open(filepath, "r") as f:
            content = f.read()
            splits[name] = json.loads(content)['data']
            split_hashes[name] = hashlib.sha256(content.encode('utf-8')).hexdigest()
            
    # Save hashes for audit
    os.makedirs("results/phase3", exist_ok=True)
    hash_file = "results/phase3/splits_hashes.json"
    if os.path.exists(hash_file):
        with open(hash_file, "r") as f:
            expected_hashes = json.load(f)
        for name in splits.keys():
            assert expected_hashes[name] == split_hashes[name], f"Split {name} has changed!"
    else:
        with open(hash_file, "w") as f:
            json.dump(split_hashes, f, indent=4)
            
    return splits

def get_batch(data_split):
    if len(data_split) == 0:
        return None, None, None
    max_m = max([s['m'] for s in data_split])
    max_k = max([s['k'] for s in data_split])
    
    batch_Bx = []
    batch_By = []
    labels = []
    for s in data_split:
        Bx = np.array(s['B_x'], dtype=np.float32)
        By = np.array(s['B_y'], dtype=np.float32)
        
        pad_Bx = np.pad(Bx, ((0, max_m - Bx.shape[0]), (0, max_k - Bx.shape[1])))
        pad_By = np.pad(By, ((0, max_m - By.shape[0]), (0, max_k - By.shape[1])))
        
        batch_Bx.append(pad_Bx)
        batch_By.append(pad_By)
        labels.append(float(s['label']))
        
    return torch.tensor(np.array(batch_Bx)), torch.tensor(np.array(batch_By)), torch.tensor(labels)

def get_metrics(y_true, y_prob):
    y_pred = (y_prob > 0.5).astype(float)
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    mac_f1 = f1_score(y_true, y_pred, average='macro')
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1])
    
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    
    brier = brier_score_loss(y_true, y_prob)
    
    return {
        'Accuracy': acc,
        'Balanced_Acc': bacc,
        'Macro_F1': mac_f1,
        'Class2_P': precision[0], 'Class2_R': recall[0], 'Class2_F1': f1[0],
        'Class3_P': precision[1], 'Class3_R': recall[1], 'Class3_F1': f1[1],
        'ROC_AUC': roc_auc,
        'PR_AUC': pr_auc,
        'Brier_Error': brier
    }

def init_model(name):
    if name == 'NestedDeepSets':
        return NestedDeepSets(hidden_dim=32)
    elif name == 'BipartiteEncoder':
        return PairConditionedBipartiteEncoder(hidden_dim=32)
    elif name == 'MatchingNetwork':
        return DifferentiableIncidenceMatchingNetwork(hidden_dim=32, temp=0.05)
    elif name == 'Ablation_FrozenCost':
        return DifferentiableIncidenceMatchingNetwork(hidden_dim=32, temp=0.05, use_learned_encoder=False)
    elif name == 'Ablation_NoSinkhorn':
        return DifferentiableIncidenceMatchingNetwork(hidden_dim=32, temp=0.05, use_sinkhorn=False)
    elif name == 'Ablation_FixedL1Sinkhorn':
        return DifferentiableIncidenceMatchingNetwork(hidden_dim=32, temp=0.05, use_learned_cost=False)
    elif name in ['MatchingNetwork_LabelShuffle', 'MatchingNetwork_Misalign']:
        return DifferentiableIncidenceMatchingNetwork(hidden_dim=32, temp=0.05)
    raise ValueError(f"Unknown model {name}")

def train_and_evaluate(seed, model_name, splits):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    model = init_model(model_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCEWithLogitsLoss()
    
    # Check if we should shuffle labels or misalign columns
    train_data = splits['train'].copy()
    random.shuffle(train_data) # Vary training order
    
    if 'LabelShuffle' in model_name:
        for s in train_data:
            s['label'] = random.choice([0, 1])
    if 'Misalign' in model_name:
        for s in train_data:
            m, k = s['m'], s['k']
            cp = np.random.permutation(k)
            B_y = np.array(s['B_y'])
            s['B_y'] = B_y[:, cp].tolist() # Misalign columns of By
            
    train_Bx, train_By, train_y = get_batch(train_data)
    val_Bx, val_By, val_y = get_batch(splits['validation'])
    
    best_val_score = -1.0
    best_state = None
    patience = 15
    patience_counter = 0
    
    start_time = time.time()
    
    # Tiny Batch for Overfit Test
    tiny_Bx, tiny_By, tiny_y = train_Bx[:32], train_By[:32], train_y[:32]
    
    diagnostics = []
    
    model.train()
    for ep in range(150):
        optimizer.zero_grad()
        logits = model(train_Bx, train_By)
        loss = criterion(logits, train_y)
        loss.backward()
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_Bx, val_By)
            val_loss = criterion(val_logits, val_y).item()
            
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_metrics = get_metrics(val_y.numpy(), val_probs)
            val_mac_f1 = val_metrics['Macro_F1']
            val_acc = val_metrics['Accuracy']
            
            train_probs = torch.sigmoid(logits.detach()).cpu().numpy()
            train_acc = accuracy_score(train_y.numpy(), (train_probs > 0.5).astype(float))
            
            diagnostics.append({
                'Epoch': ep, 'Train_Loss': loss.item(), 'Val_Loss': val_loss,
                'Train_Acc': train_acc, 'Val_Acc': val_acc, 'Val_Macro_F1': val_mac_f1,
                'Grad_Norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            })
            
            if val_mac_f1 > best_val_score:
                best_val_score = val_mac_f1
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
        if patience_counter >= patience:
            break
        model.train()
        
    pd.DataFrame(diagnostics).to_csv(f"results/phase3/diagnostics_{model_name}_seed{seed}.csv", index=False)
        
    train_time = time.time() - start_time
    
    # Save checkpoint
    os.makedirs(f"results/phase3/checkpoints", exist_ok=True)
    chkpt_path = f"results/phase3/checkpoints/{model_name}_seed{seed}.pt"
    torch.save(best_state, chkpt_path)
    
    model.load_state_dict(best_state)
    model.eval()
    
    results = {}
    preds_data = []
    
    test_splits = ['train', 'validation', 'iid_test', 'ood_template_test', 'ood_size_test', 'ood_cardinality_test', 'ood_composition_test']
    
    for split_name in test_splits:
        Bx, By, y = get_batch(splits[split_name])
        
        if Bx is None:
            continue
            
        inf_start = time.time()
        with torch.no_grad():
            logits = model(Bx, By)
            probs = torch.sigmoid(logits)
        inf_time = time.time() - inf_start
        
        metrics = get_metrics(y.numpy(), probs.numpy())
        metrics['Inference_Time_Sec'] = inf_time
        metrics['Train_Time_Sec'] = train_time
        results[split_name] = metrics
        
        # Save predictions
        for i, s in enumerate(splits[split_name]):
            preds_data.append({
                'Seed': seed, 'Model': model_name, 'Split': split_name,
                'Matched_Pair_ID': s['matched_pair_id'],
                'True_Label': int(y[i].item()),
                'Prob': float(probs[i].item())
            })
            
    return results, preds_data

def run_all():
    seeds = [42, 100, 2026, 777, 999]
    models = ['NestedDeepSets', 'BipartiteEncoder', 'MatchingNetwork', 'Ablation_FrozenCost', 'Ablation_NoSinkhorn', 'Ablation_FixedL1Sinkhorn', 'MatchingNetwork_LabelShuffle', 'MatchingNetwork_Misalign']
    
    splits = load_data()
    all_results = []
    all_preds = []
    
    os.makedirs("results/phase3", exist_ok=True)
    
    for seed in seeds:
        print(f"--- Running Seed {seed} ---")
        for model_name in models:
            print(f"Evaluating {model_name}...")
            res, preds_data = train_and_evaluate(seed, model_name, splits)
            all_preds.extend(preds_data)
            
            for split_name, metrics in res.items():
                row = {'Seed': seed, 'Model': model_name, 'Split': split_name}
                row.update(metrics)
                all_results.append(row)
                
    df = pd.DataFrame(all_results)
    df.to_csv("results/phase3/per_seed_metrics.csv", index=False)
    
    pd.DataFrame(all_preds).to_csv("results/phase3/per_sample_predictions.csv", index=False)
    
    agg = df.groupby(['Model', 'Split']).agg(['mean', 'std']).reset_index()
    agg.columns = ['_'.join(col).strip('_') for col in agg.columns.values]
    agg.to_csv("results/phase3/aggregate_metrics.csv", index=False)
    print("Saved metrics and checkpoints to results/phase3/")

if __name__ == "__main__":
    run_all()
