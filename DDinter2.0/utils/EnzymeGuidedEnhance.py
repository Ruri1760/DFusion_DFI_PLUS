import torch
import torch.nn as nn
import torch.nn.functional as F

class EnzymeGuidedEnhance(nn.Module):

    def __init__(self, hidden_dim=256, num_enzyme=5, num_heads=4, dropout=0.1, enzyme_dim = 480):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_enzyme = num_enzyme
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads

        self.enzyme_proj = nn.Linear(enzyme_dim, hidden_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.scale = head_dim ** -0.5

    def forward(self, mol_feat, enzyme_feat, mask=None):

        B, L, D = mol_feat.shape
        E = self.num_enzyme


        enzyme_base = self.enzyme_proj(enzyme_feat)   # [E, D]

        q = self.q_proj(mol_feat)      # [B, L, D]
        k = self.k_proj(enzyme_base)   # [E, D]
        v = self.v_proj(enzyme_base)   # [E, D]

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B, L, E]

        if mask is not None:

            mask = mask.unsqueeze(-1)  # [B, L, 1]
            attn = attn.masked_fill(~mask, -1e9)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        enzyme_agg = torch.matmul(attn, v)  # [B, L, D]

        gate_weight = self.gate(torch.cat([mol_feat, enzyme_agg], dim=-1))
        enhanced = mol_feat + gate_weight * enzyme_agg

        enhanced = self.norm(enhanced)
        enhanced = self.out_proj(enhanced)

        return enhanced, attn
