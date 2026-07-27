import json
import os

for nb_file in ['Phase3_Research.ipynb', 'Phase3_Explained_FA.ipynb']:
    if not os.path.exists(nb_file): continue
    with open(nb_file, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            lines = cell['source']
            new_lines = []
            for line in lines:
                new_line = line.replace("agg_df[['model_name', 'split'", "agg_df[['Model', 'Split'")
                if new_line != line:
                    modified = True
                new_lines.append(new_line)
            cell['source'] = new_lines
            
    if modified:
        with open(nb_file, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f'Fixed agg_df columns in {nb_file}')
