import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAggregator(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.1, alpha=0.2):
        super(GlobalAggregator, self).__init__()
        self.hidden_size = hidden_size

        # attention over (query=item embedding, key=neighbor embedding, edge weight)
        self.w_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_edge = nn.Linear(1, hidden_size, bias=False)  # lifts scalar co-occurrence weight into hidden space
        self.attn_vec = nn.Linear(hidden_size, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(alpha)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, item_ids, item_embed_table, neighbor_idx, neighbor_weight):
        query_embed = item_embed_table[item_ids]          # [B, N, D]
        nbr_ids = neighbor_idx[item_ids]                   # [B, N, K]
        nbr_w = neighbor_weight[item_ids]                  # [B, N, K]
        nbr_embed = item_embed_table[nbr_ids]              # [B, N, K, D]

        q = self.w_query(query_embed).unsqueeze(2)         # [B, N, 1, D]
        k = self.w_key(nbr_embed)                          # [B, N, K, D]
        e = self.w_edge(nbr_w.unsqueeze(-1))                # [B, N, K, D]

        score = self.attn_vec(self.leaky_relu(q + k + e)).squeeze(-1)  # [B, N, K]

        # mask out padding neighbor slots (weight 0 = "no neighbor here") so they
        # never pull attention mass away from real neighbors
        pad_mask = (nbr_w == 0)
        score = score.masked_fill(pad_mask, float('-inf'))
        no_neighbors = pad_mask.all(dim=-1, keepdim=True)
        score = torch.where(no_neighbors, torch.zeros_like(score), score)

        attn = F.softmax(score, dim=-1)                     # [B, N, K]
        attn = attn.masked_fill(pad_mask, 0.0)

        h_global = torch.sum(attn.unsqueeze(-1) * nbr_embed, dim=2)  # [B, N, D]
        h_global = self.dropout(h_global)
        return h_global


class GatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super(GatedFusion, self).__init__()
        self.gate = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, h_local, h_global):
        g = torch.sigmoid(self.gate(torch.cat([h_local, h_global], dim=-1)))
        return g * h_local + (1 - g) * h_global