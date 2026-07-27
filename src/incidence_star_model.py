import torch
import torch.nn as nn
import torch.nn.functional as F

class NestedDeepSets(nn.Module):
    """
    Model A: Baseline nested DeepSets.
    Processes B_x and B_y independently, then aggregates symmetrically.
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Since neighbors have no features, we just use a constant or degree feature
        # But wait! We need to embed vertices. 
        # Actually, let's use a simple learnable embedding per neighbor column? NO, breaks invariance.
        # We'll just pass a constant feature 1.0 for each neighbor to show it fails!
        self.node_mlp = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.edge_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.star_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        
        self.classifier = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        
    def encode_star(self, B):
        # B is shape (batch, m, k)
        # We treat each 1 as an occurrence of a node.
        # Node feature is just 1.0
        batch_size, m, k = B.shape
        node_feats = torch.ones((batch_size, m, k, 1), device=B.device)
        node_embs = self.node_mlp(node_feats) # (batch, m, k, hidden)
        
        # Mask out zeros in B
        mask = B.unsqueeze(-1) # (batch, m, k, 1)
        masked_embs = node_embs * mask
        
        # Sum over neighbors (k)
        edge_sum = masked_embs.sum(dim=2) # (batch, m, hidden)
        edge_embs = self.edge_mlp(edge_sum)
        
        # Sum over hyperedges (m)
        star_sum = edge_embs.sum(dim=1) # (batch, hidden)
        star_embs = self.star_mlp(star_sum)
        return star_embs

    def forward_direction(self, Bx, By):
        pair_rep = self.get_embeddings(Bx, By)
        return self.classifier(pair_rep).squeeze(-1)

    def forward(self, Bx, By):
        return 0.5 * (self.forward_direction(Bx, By) + self.forward_direction(By, Bx))
        
    def get_embeddings(self, Bx, By):
        hx = self.encode_star(Bx)
        hy = self.encode_star(By)
        pair_rep = torch.cat([hx + hy, torch.abs(hx - hy), hx * hy], dim=1)
        return pair_rep


class PairConditionedBipartiteEncoder(nn.Module):
    """
    Model B: Pair-Conditioned Bipartite Encoder.
    Processes B_x and B_y jointly over shared neighbors.
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.neighbor_init = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        self.msg_x = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.msg_y = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        
        self.update_n = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        
        self.classifier = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        
    def get_embeddings(self, Bx, By):
        batch_size, mx, k = Bx.shape
        _, my, _ = By.shape
        
        # Initialize neighbor embeddings
        n_embs = self.neighbor_init.expand(batch_size, k, -1) # (batch, k, hidden)
        
        # 1. Neighbor -> Hyperedge (DeepSet over neighbors)
        msg_to_x = torch.bmm(Bx, n_embs) # (batch, mx, hidden)
        msg_to_y = torch.bmm(By, n_embs) # (batch, my, hidden)
        
        x_edge_embs = self.msg_x(msg_to_x)
        y_edge_embs = self.msg_x(msg_to_y) # Shared!
        
        # 2. Hyperedge -> Neighbor (DeepSet over incident edges)
        msg_from_x = torch.bmm(Bx.transpose(1, 2), x_edge_embs) # (batch, k, hidden)
        msg_from_y = torch.bmm(By.transpose(1, 2), y_edge_embs) # (batch, k, hidden)
        
        # Update neighbor embeddings jointly
        n_cat = torch.cat([n_embs, msg_from_x, msg_from_y], dim=-1)
        n_embs_new = self.update_n(n_cat) # (batch, k, hidden)
        
        # Final aggregation over neighbors to get pair representation
        hx = msg_from_x.sum(dim=1)
        hy = msg_from_y.sum(dim=1)
        
        pair_rep = torch.cat([hx + hy, torch.abs(hx - hy), hx * hy], dim=1)
        return pair_rep

    def forward_direction(self, Bx, By):
        pair_rep = self.get_embeddings(Bx, By)
        return self.classifier(pair_rep).squeeze(-1)

    def forward(self, Bx, By):
        return 0.5 * (self.forward_direction(Bx, By) + self.forward_direction(By, Bx))
