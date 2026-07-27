import os
import json
import hashlib

def file_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def generate_final_manifest():
    manifest = {
        'run_manifest_sha256': file_sha256('results/phase3/run_manifest.json'),
        'Phase3_Research.ipynb_sha256': file_sha256('Phase3_Research.ipynb'),
        'Phase3_Explained_FA.ipynb_sha256': file_sha256('Phase3_Explained_FA.ipynb'),
        'architecture_svg_sha256': file_sha256('results/phase3/figures/model_architecture.svg'),
        'architecture_png_sha256': file_sha256('results/phase3/figures/model_architecture.png'),
    }
    
    with open('results/phase3/final_artifact_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=4)
        
    print("Generated final_artifact_manifest.json")

if __name__ == '__main__':
    generate_final_manifest()
