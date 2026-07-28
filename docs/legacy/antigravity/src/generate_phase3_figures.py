import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_figures():
    os.makedirs("results/phase3/figures", exist_ok=True)
    
    # 1. Load Data
    try:
        df = pd.read_csv("results/phase3/per_seed_metrics.csv")
    except Exception as e:
        print("Data not found yet, skipping figures.")
        return
        
    # Set style
    sns.set_theme(style="whitegrid")
    
    # 2. Model comparison with Mean +/- Std
    # Compare Test Acc across models on ood_template_test
    plt.figure(figsize=(10, 6))
    ood_data = df[df['Split'] == 'ood_template_test']
    sns.barplot(data=ood_data, x='Model', y='Accuracy', capsize=.1, errorbar='sd')
    plt.title('OOD Template Test Accuracy by Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("results/phase3/figures/model_comparison.png")
    plt.close()
    
    # 3. IID versus OOD
    # For MatchingNetwork, compare across all splits
    plt.figure(figsize=(12, 6))
    mn_data = df[df['Model'] == 'MatchingNetwork']
    sns.barplot(data=mn_data, x='Split', y='Accuracy', capsize=.1, errorbar='sd')
    plt.title('Matching Network: Generalization Across Splits')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("results/phase3/figures/iid_vs_ood.png")
    plt.close()
    
    # 4. Ablations
    plt.figure(figsize=(10, 6))
    ablation_models = ['MatchingNetwork', 'Ablation_FrozenCost', 'Ablation_NoSinkhorn', 'Ablation_FixedL1Sinkhorn']
    ab_data = df[(df['Model'].isin(ablation_models)) & (df['Split'] == 'ood_template_test')]
    sns.barplot(data=ab_data, x='Model', y='Accuracy', capsize=.1, errorbar='sd')
    plt.title('Ablation Study on OOD Template Test')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("results/phase3/figures/ablation_comparison.png")
    plt.close()
    
    print("Generated basic figures.")

if __name__ == "__main__":
    plot_figures()
