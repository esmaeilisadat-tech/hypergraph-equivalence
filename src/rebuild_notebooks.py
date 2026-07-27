import os
import subprocess
import json

def validate_executed_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            for out in cell.get('outputs', []):
                if out.get('output_type') == 'error':
                    return False, f"Execution error in cell: {out.get('evalue')}"
                    
        # Check for stale values
        source = "".join(cell.get('source', []))
        stale_strings = ['0.62775', '0.62450', '0.50600', '1536 errors', 'C_raw + C_neural', 'solves the entire problem']
        for stale in stale_strings:
            if stale in source:
                return False, f"Stale value '{stale}' found in notebook!"
                
    return True, "Valid"

def rebuild_and_validate(notebook_name):
    # Processed in update_notebooks.py as .rebuilt.ipynb
    rebuilt_name = notebook_name.replace('.ipynb', '.rebuilt.ipynb')
    executed_name = notebook_name.replace('.ipynb', '.executed.ipynb')
    
    print(f"Executing {rebuilt_name}...")
    cmd = [
        "jupyter", "nbconvert", "--to", "notebook", "--execute", rebuilt_name,
        "--output", executed_name, "--ExecutePreprocessor.timeout=1800"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED to execute {notebook_name}:\n{result.stderr}")
        return False
        
    print(f"Validating {executed_name}...")
    is_valid, msg = validate_executed_notebook(executed_name)
    if not is_valid:
        print(f"VALIDATION FAILED for {notebook_name}: {msg}")
        return False
        
    print(f"Validation passed. Replacing {notebook_name}...")
    os.replace(executed_name, notebook_name)
    os.remove(rebuilt_name)
    return True

if __name__ == '__main__':
    ok1 = rebuild_and_validate('Phase3_Research.ipynb')
    ok2 = rebuild_and_validate('Phase3_Explained_FA.ipynb')
    
    if ok1 and ok2:
        print("Both notebooks successfully rebuilt and validated.")
    else:
        print("Notebook rebuild failed.")
