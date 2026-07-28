import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_architecture():
    os.makedirs('results/phase3/figures', exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Define models
    boxes = [
        ("NestedDeepSets\n(Set-to-Set)", 0.1, 0.6),
        ("BipartiteEncoder\n(Node/Hyperedge)", 0.5, 0.6),
        ("NoSinkhorn Ablation\n(Dot Product)", 0.1, 0.2),
        ("Fixed-L1 Sinkhorn\n(Batched OT)", 0.5, 0.2)
    ]
    
    for text, x, y in boxes:
        rect = patches.Rectangle((x, y), 0.3, 0.2, linewidth=2, edgecolor='black', facecolor='lightblue', zorder=2)
        ax.add_patch(rect)
        ax.text(x+0.15, y+0.1, text, ha='center', va='center', fontsize=12, fontweight='bold', zorder=3)
        
    # Arrows
    ax.annotate('', xy=(0.45, 0.7), xytext=(0.4, 0.7), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate('', xy=(0.25, 0.6), xytext=(0.25, 0.4), arrowprops=dict(arrowstyle="->", lw=2, color='gray', ls='--'))
    ax.annotate('', xy=(0.65, 0.6), xytext=(0.65, 0.4), arrowprops=dict(arrowstyle="->", lw=2, color='gray', ls='--'))
    
    plt.title("Four-Model Architecture in Phase 3", fontsize=16)
    
    plt.savefig('results/phase3/figures/model_architecture.svg', format='svg', bbox_inches='tight')
    plt.savefig('results/phase3/figures/model_architecture.png', format='png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == '__main__':
    draw_architecture()
