import json
import os

def generate_configs():
    os.makedirs("results/phase3/model_configs", exist_ok=True)
    
    base_config = {
        "hidden_dim": 32,
        "beta": 5.0,
        "tau_start": 0.5,
        "tau_final": 0.05,
        "train_iters": 50,
        "eval_iters": 500,
        "cost_normalization": "feature_dimension",
        "selected_threshold": 0.5
    }
    
    # 1. Aggregate MLP Baseline (No Learned Encoder, No Sinkhorn)
    config_mlp = base_config.copy()
    config_mlp.update({
        "model_name": "MLP_Baseline",
        "learned_encoder": False,
        "learned_residual_cost": False,
        "sinkhorn_enabled": False,
        "straight_through_hungarian": False
    })
    
    # 2. NestedDeepSets
    config_nds = base_config.copy()
    config_nds.update({
        "model_name": "NestedDeepSets",
        "learned_encoder": True,
        "learned_residual_cost": False,
        "sinkhorn_enabled": False,
        "straight_through_hungarian": False
    })
    
    # 3. BipartiteEncoder
    config_bp = base_config.copy()
    config_bp.update({
        "model_name": "BipartiteEncoder",
        "learned_encoder": True,
        "learned_residual_cost": False,
        "sinkhorn_enabled": False,
        "straight_through_hungarian": False
    })
    
    # 4. Final MatchingNetwork (Fixed-L1 Sinkhorn)
    config_final = base_config.copy()
    config_final.update({
        "model_name": "MatchingNetwork",
        "learned_encoder": True,
        "learned_residual_cost": False,
        "sinkhorn_enabled": True,
        "straight_through_hungarian": False
    })
    
    # 5. Learned-cost Sinkhorn (Ablation)
    config_learned_cost = base_config.copy()
    config_learned_cost.update({
        "model_name": "LearnedCostSinkhorn",
        "learned_encoder": True,
        "learned_residual_cost": True,
        "sinkhorn_enabled": True,
        "straight_through_hungarian": False
    })
    
    # 6. Straight-through Hungarian (Ablation)
    config_st = base_config.copy()
    config_st.update({
        "model_name": "StraightThroughHungarian",
        "learned_encoder": True,
        "learned_residual_cost": False,
        "sinkhorn_enabled": False,
        "straight_through_hungarian": True
    })
    
    # 7. No-Sinkhorn ablation
    config_no_sinkhorn = base_config.copy()
    config_no_sinkhorn.update({
        "model_name": "NoSinkhorn",
        "learned_encoder": True,
        "learned_residual_cost": False,
        "sinkhorn_enabled": False,
        "straight_through_hungarian": False
    })
    
    configs = [
        config_mlp,
        config_nds,
        config_bp,
        config_final,
        config_learned_cost,
        config_st,
        config_no_sinkhorn
    ]
    
    for cfg in configs:
        name = cfg["model_name"]
        path = f"results/phase3/model_configs/{name}_config.json"
        with open(path, "w") as f:
            json.dump(cfg, f, indent=4)
            
    # Also save the final one as final_model_config.json
    with open("results/phase3/model_configs/final_model_config.json", "w") as f:
        json.dump(config_final, f, indent=4)
        
    print("Generated model configurations.")

def load_config(model_name):
    path = f"results/phase3/model_configs/{model_name}_config.json"
    with open(path, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    generate_configs()
