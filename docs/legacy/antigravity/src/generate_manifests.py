import os
import json
import hashlib

def hash_file(filepath):
    if not os.path.exists(filepath):
        return None
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def generate_manifests():
    phase3_dir = "results/phase3"
    
    run_manifest = {
        "checkpoints": {},
        "predictions": {},
        "metrics": {}
    }
    
    for chkpt in os.listdir(os.path.join(phase3_dir, "checkpoints")):
        if chkpt.endswith(".pt"):
            run_manifest["checkpoints"][chkpt] = hash_file(os.path.join(phase3_dir, "checkpoints", chkpt))
            
    for pred in os.listdir(os.path.join(phase3_dir, "predictions")):
        if pred.endswith(".csv"):
            run_manifest["predictions"][pred] = hash_file(os.path.join(phase3_dir, "predictions", pred))
            
    for metric in ["aggregate_metrics.csv", "per_seed_metrics.csv", "numerical_audit.json", "split_overlap_audit.json"]:
        run_manifest["metrics"][metric] = hash_file(os.path.join(phase3_dir, metric))
        
    with open(os.path.join(phase3_dir, "run_manifest.json"), "w") as f:
        json.dump(run_manifest, f, indent=4)
        
    final_artifact_manifest = {
        "run_manifest.json": hash_file(os.path.join(phase3_dir, "run_manifest.json")),
        "final_validation_report.md": hash_file(os.path.join(phase3_dir, "final_validation_report.md")),
        "notebooks": {}
    }
    
    if os.path.exists("notebooks"):
        for nb in os.listdir("notebooks"):
            if nb.endswith(".ipynb"):
                final_artifact_manifest["notebooks"][nb] = hash_file(os.path.join("notebooks", nb))
                
    with open(os.path.join(phase3_dir, "final_artifact_manifest.json"), "w") as f:
        json.dump(final_artifact_manifest, f, indent=4)
        
if __name__ == "__main__":
    generate_manifests()
