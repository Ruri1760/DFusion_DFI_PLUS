# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class EnzymeGuidedEnhance(nn.Module):
#     """
#     创新点：
#     1. 构建可学习CYP450酶原型库，不直接用ESM输出当固定Query
#     2. 门控机制自适应筛选分子该吸收哪种酶的生物信息
#     3. 双向相似度加权 + 特征门控融合，避开原有伪节点交叉注意力结构
#     4. 输入输出维度和原模块完全一致，下游预测层不用改
#     """
#     def __init__(self, hidden_dim=256, num_enzyme=5, num_heads=4, dropout=0.1, enzyme_dim = 480):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.num_enzyme = num_enzyme
#         self.num_heads = num_heads
#         head_dim = hidden_dim // num_heads

#         # 可学习酶原型偏移：在ESM预训练酶基础上再微调
#         self.enzyme_proj = nn.Linear(enzyme_dim, hidden_dim)

#         # 分子→酶 双向相似度映射
#         self.q_proj = nn.Linear(hidden_dim, hidden_dim)
#         self.k_proj = nn.Linear(hidden_dim, hidden_dim)
#         self.v_proj = nn.Linear(hidden_dim, hidden_dim)

#         # 门控自适应融合：控制酶特征流入比例
#         self.gate = nn.Sequential(
#             nn.Linear(2 * hidden_dim, hidden_dim),
#             nn.Sigmoid()
#         )

#         # 输出融合 & 归一化
#         self.out_proj = nn.Linear(hidden_dim, hidden_dim)
#         self.norm = nn.LayerNorm(hidden_dim)
#         self.dropout = nn.Dropout(dropout)

#         self.scale = head_dim ** -0.5

#     def forward(self, mol_feat, enzyme_feat):
#         """
#         mol_feat: 分子图特征 [B, 1, D]
#         enzyme_feat: CYP450 酶特征 [5, D]
#         return: 酶增强后分子特征 [B, 1, D]
#         """
#         B, _, D = mol_feat.shape
#         E = self.num_enzyme

#         # 1. 构建可学习酶原型（创新：在ESM特征上加可学习偏移）
#         enzyme_base = self.enzyme_proj(enzyme_feat)   # [5, D]

#         # 2. 投影 Q K V
#         q = self.q_proj(mol_feat)      # [B,1,D]
#         k = self.k_proj(enzyme_base)   # [5,D]
#         v = self.v_proj(enzyme_base)   # [5,D]

#         # 3. 分子与所有酶计算相似度权重
#         attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B,1,5]
#         attn = F.softmax(attn, dim=-1)
#         attn = self.dropout(attn)

#         # 4. 加权聚合所有酶的生物特征
#         enzyme_agg = torch.matmul(attn, v)  # [B,1,D]

#         # 5. 门控自适应融合（核心创新：控制酶信息融入多少）
#         gate_weight = self.gate(torch.cat([mol_feat, enzyme_agg], dim=-1))
#         enhanced = mol_feat + gate_weight * enzyme_agg

#         # 6. 归一化与输出
#         enhanced = self.norm(enhanced)
#         enhanced = self.out_proj(enhanced)
#         return enhanced, attn
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class EnzymeGuidedEnhance(nn.Module):
    """
    适配变长原子序列版本：
    输入 mol_feat: [B, L, D]  原子级特征 (L为动态原子数)
    支持 mask: [B, L] 有效原子掩码
    输出 enhanced_feat: [B, L, D] 原子增强特征
    输出 attn: [B, L, num_enzyme]  原子-酶注意力权重（可解释位点）
    原有全部创新点：可学习酶原型、相似度加权、门控融合 完整保留
    """
    def __init__(self, hidden_dim=256, num_enzyme=5, num_heads=4, dropout=0.1, enzyme_dim = 480):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_enzyme = num_enzyme
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads

        # 可学习酶原型投影（原逻辑不变）
        self.enzyme_proj = nn.Linear(enzyme_dim, hidden_dim)

        # Q/K/V 投影
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # 特征门控融合
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        # 输出层 & 归一化
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.scale = head_dim ** -0.5

    def forward(self, mol_feat, enzyme_feat, mask=None):
        """
        Args:
            mol_feat: 原子特征 [B, L, D]
            enzyme_feat: 固定酶特征 [num_enzyme, enzyme_dim]
            mask: 有效原子掩码 [B, L], True=真实原子, False=padding
        Returns:
            enhanced: [B, L, D] 增强后原子特征
            attn: [B, L, num_enzyme]  每个原子对5个酶的注意力权重（可解释）
        """
        B, L, D = mol_feat.shape
        E = self.num_enzyme

        # 1. 可学习酶原型 (完全沿用原逻辑)
        enzyme_base = self.enzyme_proj(enzyme_feat)   # [E, D]

        # 2. Q/K/V 投影
        q = self.q_proj(mol_feat)      # [B, L, D]
        k = self.k_proj(enzyme_base)   # [E, D]
        v = self.v_proj(enzyme_base)   # [E, D]

        # 3. 原子 <-> 酶 相似度注意力 [B, L, E]
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B, L, E]

        # ========== 新增：掩码屏蔽padding区域 ==========
        if mask is not None:
            # padding位置设为 -inf，softmax后权重趋近0
            mask = mask.unsqueeze(-1)  # [B, L, 1]
            attn = attn.masked_fill(~mask, -1e9)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 4. 用酶特征加权增强每个原子 [B, L, D]
        enzyme_agg = torch.matmul(attn, v)  # [B, L, D]

        # 5. 门控融合（原核心逻辑不变，逐原子执行）
        gate_weight = self.gate(torch.cat([mol_feat, enzyme_agg], dim=-1))
        enhanced = mol_feat + gate_weight * enzyme_agg

        # 6. 归一化 & 输出
        enhanced = self.norm(enhanced)
        enhanced = self.out_proj(enhanced)

        return enhanced, attn