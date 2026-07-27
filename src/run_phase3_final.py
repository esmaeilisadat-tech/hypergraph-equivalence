import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import hashlib
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score, average_precision_score, brier_score_loss

from src.differentiable_matching import DifferentiableIncidenceMatchingNetwork, NestedDeepSets, PairConditionedBipartiteEncoder
from src.oracles import exact_counter_oracle, hungarian_matching_oracle
from src.config_manager import load_config
from src.audit_splits import get_hashes

def load_data():
    splits = {}
    for name in ['train', 'validation', 'iid_test', 'ood_template_test', 'ood_size_test']:
        filepath = f"data/splits/{name}.json"
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                splits[name] = json.loads(f.read())['data']
    return splits

def get_batch(data_split):
    if len(data_split) == 0:
        return None, None, None
    max_m = max([s['m'] for s in data_split])
    max_k = max([s['k'] for s in data_split])
    
    batch_Bx, batch_By, labels = [], [], []
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
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    brier = brier_score_loss(y_true, y_prob)
    return {
        'Accuracy': acc, 'Balanced_Acc': bacc, 'Macro_F1': mac_f1,
        'Class2_P': precision[0], 'Class2_R': recall[0], 'Class2_F1': f1[0],
        'Class3_P': precision[1], 'Class3_R': recall[1], 'Class3_F1': f1[1],
        'ROC_AUC': roc_auc, 'PR_AUC': pr_auc, 'Brier_Error': brier
    }

def init_model(model_name):
    cfg = load_config(model_name)
    if model_name == "NestedDeepSets":
        return NestedDeepSets(hidden_dim=cfg["hidden_dim"])
    elif model_name == "BipartiteEncoder":
        return PairConditionedBipartiteEncoder(hidden_dim=cfg["hidden_dim"])
    else:
        # MLP_Baseline, MatchingNetwork, LearnedCostSinkhorn, StraightThroughHungarian, NoSinkhorn
        return DifferentiableIncidenceMatchingNetwork(cfg)

def train_model(model, Bx, By, y, val_Bx=None, val_By=None, val_y=None, max_epochs=200, patience=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val = -1.0
    best_state = None
    pat_count = 0
    
    # Get temperature schedule from config or defaults
    has_temp = hasattr(model, 'temp')
    if has_temp:
        tau_start = getattr(model, 'config', {}).get("tau_start", 0.5)
        tau_final = getattr(model, 'config', {}).get("tau_final", 0.05)
        decay_eps = int(max_epochs * 0.7)
    
    for ep in range(max_epochs):
        if has_temp:
            if ep < decay_eps:
                model.temp = tau_start - (tau_start - tau_final) * (ep / decay_eps)
            else:
                model.temp = tau_final
                
        model.train()
        optimizer.zero_grad()
        logits = model(Bx, By)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            model.eval()
            old_iters = getattr(model, 'eval_iters', None)
            if old_iters is not None:
                model.eval_iters = getattr(model, 'train_iters', 50)
                
            if val_Bx is not None:
                val_logits = model(val_Bx, val_By)
                val_loss = criterion(val_logits, val_y)
                val_pred = (val_logits > 0).float()
                val_acc = (val_pred == val_y).float().mean().item()
                
                if val_acc >= best_val:
                    best_val = val_acc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    pat_count = 0
                else:
                    pat_count += 1
                    
                if pat_count >= patience:
                    break
            else:
                train_pred = (logits > 0).float()
                val_acc = (train_pred == y).float().mean().item()
                if val_acc >= best_val:
                    best_val = val_acc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
            if old_iters is not None:
                model.eval_iters = old_iters
                
    if best_state is not None and val_Bx is not None:
        model.load_state_dict(best_state)
        
    model.eval()
    with torch.no_grad():
        train_probs = torch.sigmoid(model(Bx, By)).cpu().numpy()
        train_acc = accuracy_score(y.numpy(), (train_probs > 0.5).astype(float))
        
    return model, train_acc

def hash_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def gate_a(splits):
    print("--- GATE A: Oracle Verification ---")
    data = splits['validation']
    c_correct, h_correct = 0, 0
    per_sample = []
    
    for s in data:
        Bx, By = np.array(s['B_x']), np.array(s['B_y'])
        y = float(s['label'])
        c_pred = exact_counter_oracle(Bx, By)
        h_pred = hungarian_matching_oracle(Bx, By)
        if c_pred == y: c_correct += 1
        if h_pred == y: h_correct += 1
        per_sample.append({
            "sample_id": s.get('sample_id', get_hashes(s)[4]),
            "y_true": y,
            "counter_pred": c_pred,
            "hungarian_pred": h_pred
        })
        
    c_acc = c_correct / len(data)
    h_acc = h_correct / len(data)
    
    os.makedirs("results/phase3/oracle_predictions", exist_ok=True)
    pd.DataFrame(per_sample).to_csv("results/phase3/oracle_predictions/gate_a_oracles.csv", index=False)
    
    assert c_acc == 1.0, f"Gate A Failed: Counter Oracle Acc = {c_acc}"
    assert h_acc == 1.0, f"Gate A Failed: Hungarian Oracle Acc = {h_acc}"
    print("Gate A PASSED")

def gate_b(splits):
    print("--- GATE B: Baseline Health & Robustness ---")
    data = splits['train']
    c2 = [s for s in data if s['label'] == 0][:32]
    c3 = [s for s in data if s['label'] == 1][:32]
    overfit_data = c2 + c3
    Bx, By, y = get_batch(overfit_data)
    
    baselines = ['MLP_Baseline', 'NestedDeepSets', 'BipartiteEncoder', 'LearnedCostSinkhorn', 'StraightThroughHungarian', 'NoSinkhorn', 'MatchingNetwork']
    invalid_models = []
    
    for variant in baselines:
        seed = 42
        torch.manual_seed(seed)
        model = init_model(variant)
        model, train_acc = train_model(model, Bx, By, y, max_epochs=300, patience=300) # train until convergence
        if train_acc < 0.99:
            print(f"  {variant} FAILED Gate B1 (acc={train_acc}). Marked as INVALID_FOR_COMPARISON.")
            invalid_models.append(variant)
        else:
            print(f"  {variant} PASSED Gate B1.")
            
        if variant == 'MatchingNetwork':
            # Gate B2
            for s in [42, 100, 2026, 777, 999]:
                torch.manual_seed(s)
                m = init_model(variant)
                m, ta = train_model(m, Bx, By, y, max_epochs=500, patience=500)
                assert ta >= 0.99, f"Gate B2 Failed: Final model seed {s} got {ta}"
            print("  MatchingNetwork PASSED Gate B2 (all seeds >= 99%).")
            
    print("Gate B PASSED")
    return invalid_models

def checkpoint_roundtrip_test(model, model_name, Bx, By, seed):
    print("  Running checkpoint roundtrip test...")
    model.eval()
    model.float()
    Bx_f, By_f = Bx.float(), By.float()
    
    with torch.no_grad():
        logits1 = model(Bx_f, By_f)
        probs1 = torch.sigmoid(logits1)
        
    chkpt_path = f"results/phase3/checkpoints/roundtrip_test_{seed}.pt"
    torch.save(model.state_dict(), chkpt_path)
    chkpt_hash = hash_file(chkpt_path)
    
    model2 = init_model(model_name)
    model2.load_state_dict(torch.load(chkpt_path))
    model2.eval()
    model2.float()
    
    with torch.no_grad():
        logits2 = model2(Bx_f, By_f)
        probs2 = torch.sigmoid(logits2)
        
    diff_logit = torch.max(torch.abs(logits1 - logits2)).item()
    diff_prob = torch.max(torch.abs(probs1 - probs2)).item()
    
    if diff_logit > 1e-10: print(f"Warning: Roundtrip logit diff {diff_logit} > 1e-10")
    if diff_prob > 1e-10: print(f"Warning: Roundtrip prob diff {diff_prob} > 1e-10")
    
    os.remove(chkpt_path)
    return diff_logit, diff_prob, chkpt_hash

def gate_c(splits):
    print("--- GATE C: Final Model Requirements ---")
    train_Bx, train_By, train_y = get_batch(splits['train'])
    val_Bx, val_By, val_y = get_batch(splits['validation'])
    
    torch.manual_seed(42)
    model = init_model('MatchingNetwork')
    model, train_acc = train_model(model, train_Bx, train_By, train_y, val_Bx, val_By, val_y, max_epochs=300, patience=30)
    
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(train_Bx, train_By)).cpu().numpy()
    
    metrics = get_metrics(train_y.numpy(), probs)
    print(f"Gate C Metrics: Train Acc {metrics['Accuracy']:.3f}, Balanced Acc {metrics['Balanced_Acc']:.3f}, Macro F1 {metrics['Macro_F1']:.3f}, Min Recall {min(metrics['Class2_R'], metrics['Class3_R']):.3f}")
    
    assert metrics['Accuracy'] >= 0.95, "Failed Gate C: Train Accuracy < 95%"
    assert metrics['Balanced_Acc'] >= 0.95, "Failed Gate C: Train Balanced Accuracy < 95%"
    assert metrics['Macro_F1'] >= 0.95, "Failed Gate C: Train Macro F1 < 95%"
    assert metrics['Class2_R'] >= 0.90 and metrics['Class3_R'] >= 0.90, "Failed Gate C: Min Recall < 90%"
    
    print("Gate C PASSED")

def trace_evaluation(model_name, seed, split_name, dataset, first_10_samples, first_10_hashes, y_true, preds_path):
    trace_path = "results/phase3/evaluation_trace.jsonl"
    record = {
        "model_name": model_name,
        "seed": seed,
        "requested_split_name": split_name,
        "dataset_object_id": id(dataset),
        "sample_count": len(dataset),
        "first_10_sample_ids": first_10_samples,
        "first_10_pair_hashes": first_10_hashes,
        "label_distribution": {str(k): int(v) for k, v in zip(*np.unique(y_true, return_counts=True))},
        "template_distribution": {str(k): int(v) for k, v in zip(*np.unique([s.get('template_family', '') for s in dataset], return_counts=True))},
        "size_distribution": {str(k): int(v) for k, v in zip(*np.unique([s['m'] for s in dataset], return_counts=True))},
        "prediction_file_path": preds_path,
        "prediction_file_sha256": hash_file(preds_path) if os.path.exists(preds_path) else None
    }
    with open(trace_path, "a") as f:
        f.write(json.dumps(record) + "\n")

def run_5seed(splits, invalid_models):
    print("--- Starting Full Evaluation ---")
    seeds = [42, 100, 2026, 777, 999]
    models = ['MLP_Baseline', 'NestedDeepSets', 'BipartiteEncoder', 'LearnedCostSinkhorn', 'StraightThroughHungarian', 'NoSinkhorn', 'MatchingNetwork']
    
    train_Bx, train_By, train_y = get_batch(splits['train'])
    val_Bx, val_By, val_y = get_batch(splits['validation'])
    
    os.makedirs("results/phase3/checkpoints", exist_ok=True)
    os.makedirs("results/phase3/predictions", exist_ok=True)
    
    all_preds = []
    
    for seed in seeds:
        print(f"Seed {seed}")
        for model_name in models:
            if model_name in invalid_models:
                print(f"  Skipping {model_name} (INVALID_FOR_COMPARISON)")
                continue
                
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            model = init_model(model_name)
            chkpt_path = f"results/phase3/checkpoints/{model_name}_seed{seed}.pt"
            
            start_t = time.time()
            model, _ = train_model(model, train_Bx, train_By, train_y, val_Bx, val_By, val_y, max_epochs=300, patience=30)
            train_time = time.time() - start_t
            
            torch.save(model.state_dict(), chkpt_path)
            chkpt_hash = hash_file(chkpt_path)
            
            if model_name == 'MatchingNetwork':
                checkpoint_roundtrip_test(model, model_name, train_Bx[:10], train_By[:10], seed)
            
            model.eval()
            for split_name in ['train', 'validation', 'iid_test', 'ood_template_test', 'ood_size_test']:
                s_Bx, s_By, s_y = get_batch(splits[split_name])
                if s_Bx is None: continue
                
                inf_start = time.time()
                with torch.no_grad():
                    probs = torch.sigmoid(model(s_Bx, s_By)).cpu().numpy()
                inf_time = time.time() - inf_start
                
                # Trace and store predictions
                preds_path = f"results/phase3/predictions/{model_name}_{seed}_{split_name}.csv"
                split_preds = []
                first_10_ids = []
                first_10_hashes = []
                
                for i, s in enumerate(splits[split_name]):
                    _, _, meta_hash, pair_hash, sample_id, matched_pair_id = get_hashes(s)
                    if i < 10:
                        first_10_ids.append(sample_id)
                        first_10_hashes.append(pair_hash)
                        
                    p = {
                        'sample_id': sample_id,
                        'matched_pair_id': matched_pair_id,
                        'template_family': s.get('template_family', s.get('signature', '')),
                        'split': split_name,
                        'model_name': model_name,
                        'seed': seed,
                        'y_true': int(s_y[i].item()),
                        'probability_class3': float(probs[i]),
                        'threshold': 0.5,
                        'y_pred': int(probs[i] > 0.5),
                        'checkpoint_path': chkpt_path,
                        'checkpoint_sha256': chkpt_hash,
                        'tau_final': getattr(model, 'temp', None)
                    }
                    split_preds.append(p)
                    all_preds.append(p)
                    
                pd.DataFrame(split_preds).to_csv(preds_path, index=False)
                trace_evaluation(model_name, seed, split_name, splits[split_name], first_10_ids, first_10_hashes, s_y.numpy(), preds_path)
                
    # Recompute metrics from raw prediction files!
    print("Recomputing aggregated metrics from raw predictions...")
    pd.DataFrame(all_preds).to_csv("results/phase3/per_sample_predictions.csv", index=False)
    
    preds_df = pd.read_csv("results/phase3/per_sample_predictions.csv")
    recomputed_seeds = []
    
    for (m_name, sp, sd), group in preds_df.groupby(['model_name', 'split', 'seed']):
        y_true = group['y_true'].values
        y_prob = group['probability_class3'].values
        metrics = get_metrics(y_true, y_prob)
        row = {'Model': m_name, 'Split': sp, 'Seed': sd}
        row.update(metrics)
        recomputed_seeds.append(row)
        
    rec_seeds_df = pd.DataFrame(recomputed_seeds)
    rec_seeds_df.to_csv("results/phase3/per_seed_metrics.csv", index=False)
    
    rec_agg = rec_seeds_df.groupby(['Model', 'Split']).agg(['mean', 'std']).reset_index()
    rec_agg.columns = ['_'.join(col).strip('_') for col in rec_agg.columns.values]
    rec_agg.to_csv("results/phase3/aggregate_metrics.csv", index=False)
    
    print("Final predictions and metrics saved.")

if __name__ == "__main__":
    if os.path.exists("results/phase3/evaluation_trace.jsonl"):
        os.remove("results/phase3/evaluation_trace.jsonl")
        
    splits = load_data()
    gate_a(splits)
    invalid_models = gate_b(splits)
    gate_c(splits)
    run_5seed(splits, invalid_models)
