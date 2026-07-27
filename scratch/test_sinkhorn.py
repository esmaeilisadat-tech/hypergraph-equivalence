import torch
import numpy as np
import sys
sys.path.append('.')
from src.run_phase3_5seed import load_data, get_batch, init_model

splits = load_data()
Bx, By, y = get_batch(splits['ood_template_test'])
idx = np.where(y.numpy() == 1)[0][0]

m_mn = init_model('MatchingNetwork')
m_mn.load_state_dict(torch.load('results/phase3/checkpoints/MatchingNetwork_seed42.pt', map_location='cpu'))
m_mn.eval()

Bx_i = Bx[idx:idx+1]
By_i = By[idx:idx+1]

mask_x = (Bx_i.sum(dim=-1) > 0).float()
mask_y = (By_i.sum(dim=-1) > 0).float()

C = torch.randn(1, 15, 15)
C = C.abs()

P = m_mn.sinkhorn(C, mask_x, mask_y, eps=0.05, num_iters=100)

mx = mask_x.sum().item()
my = mask_y.sum().item()

row_sums = P.sum(dim=2)[0][:int(mx)]
print("mx:", mx, "my:", my)
print("expected row sums:", 1/mx)
print("actual row sums:", row_sums)
