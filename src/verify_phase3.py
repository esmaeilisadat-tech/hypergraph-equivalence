import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score, average_precision_score, brier_score_loss
import json
import hashlib
import os
import sys
sys.path.append(os.path.abspath('.'))

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

def verify():
    preds = pd.read_csv('results/phase3/per_sample_predictions.csv')
    seed_metrics = pd.read_csv('results/phase3/per_seed_metrics.csv')
    agg_metrics = pd.read_csv('results/phase3/aggregate_metrics.csv')

    recomputed_seed = []
    
    # Check per seed metrics
    print("Verifying per-seed metrics...")
    for (model, split, seed), group in preds.groupby(['model_name', 'split', 'seed']):
        y_true = group['y_true'].values
        y_prob = group['probability_class3'].values
        mets = get_metrics(y_true, y_prob)
        
        # find matching row in seed_metrics
        orig_row = seed_metrics[(seed_metrics['Model'] == model) & 
                                (seed_metrics['Split'] == split) & 
                                (seed_metrics['Seed'] == seed)]
        assert len(orig_row) == 1
        orig_row = orig_row.iloc[0]
        
        for k, v in mets.items():
            if np.isnan(v) and np.isnan(orig_row[k]): continue
            assert np.isclose(v, orig_row[k], atol=1e-5, rtol=1e-5), f"Mismatch in {k} for {model} {split} {seed}: expected {orig_row[k]}, got {v}"
            
        row = {'Model': model, 'Split': split, 'Seed': seed}
        row.update(mets)
        recomputed_seed.append(row)
        
    df_recomp = pd.DataFrame(recomputed_seed)
    
    # Check aggregate metrics
    print("Verifying aggregate metrics...")
    agg_recomp = df_recomp.groupby(['Model', 'Split']).agg(['mean', 'std']).reset_index()
    agg_recomp.columns = ['_'.join(col).strip('_') for col in agg_recomp.columns.values]
    
    for _, row in agg_recomp.iterrows():
        model = row['Model']
        split = row['Split']
        orig_row = agg_metrics[(agg_metrics['Model'] == model) & (agg_metrics['Split'] == split)]
        assert len(orig_row) == 1
        orig_row = orig_row.iloc[0]
        
        for k in row.keys():
            if k in ['Model', 'Split', 'Seed_mean', 'Seed_std']: continue
            v1 = row[k]
            v2 = orig_row[k]
            if np.isnan(v1) and np.isnan(v2): continue
            assert np.isclose(v1, v2, atol=1e-5, rtol=1e-5), f"Mismatch in agg {k} for {model} {split}: expected {v2}, got {v1}"
            
    print("Verification PASSED.")
    
    # Generate run_manifest.json
    print("Generating run_manifest.json...")
    manifest = {}
    
    def hash_file(filepath):
        h = hashlib.sha256()
        with open(filepath, 'rb') as f:
            h.update(f.read())
        return h.hexdigest()
        
    for root, _, files in os.walk('results/phase3'):
        for file in files:
            if file.endswith('.csv') or file.endswith('.pt') or file == 'sinkhorn_audit.json' or file == 'split_manifest.json':
                path = os.path.join(root, file)
                manifest[path] = hash_file(path)
                
    with open('results/phase3/run_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=4)
        
    print("run_manifest.json generated.")

if __name__ == '__main__':
    verify()
