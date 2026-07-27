import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np

def hungarian_permutation_matrix(C):
    batch_size, m_x, m_y = C.shape
    P = torch.zeros_like(C)
    for i in range(batch_size):
        c_numpy = C[i].detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(c_numpy)
        P[i, row_ind, col_ind] = 1.0
    return P

class DifferentiableIncidenceMatchingNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.get("hidden_dim", 32)
        self.beta = config.get("beta", 5.0)
        self.train_iters = config.get("train_iters", 50)
        self.eval_iters = config.get("eval_iters", 500)
        
        # Buffer for temperature so it is saved/loaded automatically
        self.register_buffer("temperature", torch.tensor(float(config.get("tau_start", 0.5))))
        
        self.learned_encoder = config.get("learned_encoder", True)
        self.learned_residual_cost = config.get("learned_residual_cost", False)
        self.sinkhorn_enabled = config.get("sinkhorn_enabled", True)
        self.straight_through_hungarian = config.get("straight_through_hungarian", False)
        
        if self.learned_encoder:
            self.neighbor_init = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
            self.elem_mlp = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim))
            self.row_mlp = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim))
        
        if self.learned_residual_cost:
            self.cost_mlp = nn.Sequential(nn.Linear(self.hidden_dim * 3, self.hidden_dim), nn.LayerNorm(self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, 1))
            self.raw_alpha = nn.Parameter(torch.tensor(0.0))
            self.alpha_max = 0.5 
            
        classifier_in = 2 + (self.hidden_dim*2 if self.learned_encoder else 0)
        self.classifier = nn.Sequential(nn.Linear(classifier_in, self.hidden_dim), nn.LayerNorm(self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, 1))

    @property
    def temp(self):
        return self.temperature.item()
        
    @temp.setter
    def temp(self, val):
        self.temperature.fill_(float(val))

    def encode_rows(self, B):
        batch_size, m, k = B.shape
        n_embs = self.neighbor_init.expand(batch_size, k, -1)
        row_embs = torch.bmm(B, n_embs)
        row_embs = self.elem_mlp(row_embs)
        row_embs = self.row_mlp(row_embs)
        return row_embs
        
    def sinkhorn(self, C, mask_x, mask_y, eps=1e-2, num_iters=50):
        assert (mask_x.sum(dim=-1) > 0).all().item(), "Found sample with empty mask_x in sinkhorn"
        assert (mask_y.sum(dim=-1) > 0).all().item(), "Found sample with empty mask_y in sinkhorn"
        
        mx = mask_x.sum(dim=-1, keepdim=True).clamp(min=1.0)
        my = mask_y.sum(dim=-1, keepdim=True).clamp(min=1.0)
        
        log_K = -C / eps
        mask_2d = mask_x.unsqueeze(-1) * mask_y.unsqueeze(1)
        log_K = log_K.masked_fill(mask_2d == 0, -1e9)
        
        log_a = torch.where(mask_x > 0, torch.log(1.0 / mx), torch.tensor(-1e9, device=C.device))
        log_b = torch.where(mask_y > 0, torch.log(1.0 / my), torch.tensor(-1e9, device=C.device))
        
        u = torch.zeros_like(log_a)
        v = torch.zeros_like(log_b)
        
        for _ in range(num_iters):
            u = torch.where(mask_x > 0, log_a - torch.logsumexp(log_K + v.unsqueeze(1), dim=2), -1e9)
            v = torch.where(mask_y > 0, log_b - torch.logsumexp(log_K.transpose(1, 2) + u.unsqueeze(1), dim=2), -1e9)
            
        log_P = log_K + u.unsqueeze(-1) + v.unsqueeze(1)
        P = torch.exp(log_P)
        P = P.masked_fill(mask_2d == 0, 0.0)
        
        P_sum = P.sum(dim=(1,2), keepdim=True)
        P_batch = torch.where(torch.abs(P_sum - 1.0) > 1e-4, P / P_sum.clamp(min=1e-12), P)
        
        return P_batch

    def forward_direction(self, Bx, By, return_aux=False):
        final_rep, aux = self.get_embeddings(Bx, By)
        logit = self.classifier(final_rep).squeeze(-1)
        if return_aux:
            return logit, aux
        return logit

    def forward(self, Bx, By, return_aux=False):
        if return_aux:
            l1, aux1 = self.forward_direction(Bx, By, return_aux=True)
            l2, aux2 = self.forward_direction(By, Bx, return_aux=True)
            return 0.5 * (l1 + l2), aux1, aux2
        return 0.5 * (self.forward_direction(Bx, By) + self.forward_direction(By, Bx))
        
    def get_embeddings(self, Bx, By):
        batch_size, mx_max, k = Bx.shape
        my_max = By.shape[1]
        
        Bx_exp = Bx.unsqueeze(2)
        By_exp = By.unsqueeze(1)
        
        mask_x = (Bx.sum(dim=-1) > 0).float()
        mask_y = (By.sum(dim=-1) > 0).float()
        
        m_x_valid = mask_x.sum(dim=1).unsqueeze(-1).clamp(min=1.0)
        m_y_valid = mask_y.sum(dim=1).unsqueeze(-1).clamp(min=1.0)
        
        # Oracle-aligned non-negative L1 base cost normalized by feature dimension k
        C_base = torch.sum(torch.abs(Bx_exp - By_exp), dim=-1) / k 
        C = C_base
        
        alpha_val = 0.0
        
        if self.learned_encoder:
            Rx = self.encode_rows(Bx)
            Ry = self.encode_rows(By)
            
            if self.learned_residual_cost:
                Rx_exp = Rx.unsqueeze(2).expand(batch_size, mx_max, my_max, self.hidden_dim)
                Ry_exp = Ry.unsqueeze(1).expand(batch_size, mx_max, my_max, self.hidden_dim)
                
                feat_diff = torch.abs(Rx_exp - Ry_exp)
                feat_sum = Rx_exp + Ry_exp
                feat_prod = Rx_exp * Ry_exp
                pair_feats = torch.cat([feat_diff, feat_sum, feat_prod], dim=-1)
                
                C_residual = self.cost_mlp(pair_feats).squeeze(-1)
                alpha_val = self.alpha_max * torch.sigmoid(self.raw_alpha)
                C = C_base + alpha_val * F.softplus(C_residual)
                
            hx = torch.sum(Rx * mask_x.unsqueeze(-1), dim=1) / m_x_valid
            hy = torch.sum(Ry * mask_y.unsqueeze(-1), dim=1) / m_y_valid
            pair_rep = torch.cat([hx + hy, torch.abs(hx - hy)], dim=-1)
        else:
            pair_rep = None
            
        aux = {'C_base': C_base, 'C': C, 'mask_x': mask_x, 'mask_y': mask_y, 'alpha': alpha_val}
            
        iters = self.train_iters if self.training else self.eval_iters
        
        if self.sinkhorn_enabled:
            P_soft = self.sinkhorn(C, mask_x, mask_y, eps=self.temperature.item(), num_iters=iters)
            aux['P'] = P_soft
            P = P_soft
            
            soft_matching_cost = torch.sum(P * C_base, dim=(1, 2)).unsqueeze(-1)
            K = torch.exp(-self.beta * C_base)
            soft_equality_score = torch.sum(P * K, dim=(1, 2)).unsqueeze(-1)
            
        elif self.straight_through_hungarian:
            P_soft = self.sinkhorn(C, mask_x, mask_y, eps=self.temperature.item(), num_iters=iters)
            
            C_masked = C.masked_fill((mask_x.unsqueeze(2) * mask_y.unsqueeze(1)) == 0, 1e9)
            P_perm = hungarian_permutation_matrix(C_masked)
            
            P_hard_batch = torch.zeros_like(P_perm)
            for i in range(batch_size):
                m_x_val = mask_x[i].sum().item()
                if m_x_val > 0:
                    P_hard_batch[i] = P_perm[i] / m_x_val
                    
            P_st = P_hard_batch.detach() - P_soft.detach() + P_soft
            P = P_st
            aux['P'] = P
            
            soft_matching_cost = torch.sum(P * C_base, dim=(1, 2)).unsqueeze(-1)
            soft_equality_score = torch.sum(P * torch.exp(-self.beta * C_base), dim=(1, 2)).unsqueeze(-1)
            
        else:
            # Baseline without Sinkhorn / Hungarian
            P = torch.zeros_like(C)
            aux['P'] = P
            # Fallback values
            soft_matching_cost = torch.zeros(batch_size, 1, device=Bx.device)
            soft_equality_score = torch.zeros(batch_size, 1, device=Bx.device)
            
        match_feats = torch.cat([soft_matching_cost, soft_equality_score], dim=-1)
            
        if pair_rep is not None:
            final_rep = torch.cat([match_feats, pair_rep], dim=-1)
        else:
            final_rep = match_feats
            
        return final_rep, aux

# Define baselines that aren't captured by DifferentiableIncidenceMatchingNetwork
class PairConditionedBipartiteEncoder(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_init = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, Bx, By, return_aux=False):
        batch_size, m_x, k_x = Bx.shape
        batch_size, m_y, k_y = By.shape
        hx = torch.bmm(Bx, self.node_init.expand(batch_size, k_x, -1))
        hy = torch.bmm(By, self.node_init.expand(batch_size, k_y, -1))
        hx = self.mlp(hx).mean(dim=1)
        hy = self.mlp(hy).mean(dim=1)
        rep = torch.cat([hx + hy, torch.abs(hx - hy)], dim=-1)
        logit = self.classifier(rep).squeeze(-1)
        if return_aux:
            return logit, {}, {}
        return logit

    def get_embeddings(self, Bx, By):
        batch_size, m_x, k_x = Bx.shape
        batch_size, m_y, k_y = By.shape
        hx = torch.bmm(Bx, self.node_init.expand(batch_size, k_x, -1))
        hy = torch.bmm(By, self.node_init.expand(batch_size, k_y, -1))
        hx = self.mlp(hx).mean(dim=1)
        hy = self.mlp(hy).mean(dim=1)
        rep = torch.cat([hx + hy, torch.abs(hx - hy)], dim=-1)
        return rep

class NestedDeepSets(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.phi = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, Bx, By, return_aux=False):
        batch_size, m_x, k_x = Bx.shape
        batch_size, m_y, k_y = By.shape
        
        hx = self.phi(Bx.unsqueeze(-1)).mean(dim=2)
        hx = self.rho(hx).mean(dim=1)
        
        hy = self.phi(By.unsqueeze(-1)).mean(dim=2)
        hy = self.rho(hy).mean(dim=1)
        
        rep = torch.cat([hx + hy, torch.abs(hx - hy)], dim=-1)
        logit = self.classifier(rep).squeeze(-1)
        if return_aux:
            return logit, {}, {}
        return logit
        
    def get_embeddings(self, Bx, By):
        batch_size, m_x, k_x = Bx.shape
        batch_size, m_y, k_y = By.shape
        hx = self.phi(Bx.unsqueeze(-1)).mean(dim=2)
        hx = self.rho(hx).mean(dim=1)
        
        hy = self.phi(By.unsqueeze(-1)).mean(dim=2)
        hy = self.rho(hy).mean(dim=1)
        
        rep = torch.cat([hx + hy, torch.abs(hx - hy)], dim=-1)
        return rep
