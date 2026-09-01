import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

from utils.multiAttention import Attention, MultiheadAttention, ImprovedMultiheadAttention
from utils.EnzymeGuidedEnhance import EnzymeGuidedEnhance

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def open_files(file_name):
    with open("./pre_params/" + file_name, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

# 全局数据加载
drug_atom_fea = open_files("drug_all_atom_fea.pkl")
food_atom_fea = open_files("food_all_atom_fea.pkl")
drug_smiles = open_files("drug_id2smiles.pkl")
food_smiles = open_files("food_id2smiles.pkl")

def get_bond_feature(bond):
    bond_type = bond.GetBondType()
    bond_type_feat = [
        1.0 if bond_type == rdchem.BondType.SINGLE else 0.0,
        1.0 if bond_type == rdchem.BondType.DOUBLE else 0.0,
        1.0 if bond_type == rdchem.BondType.TRIPLE else 0.0,
        1.0 if bond_type == rdchem.BondType.AROMATIC else 0.0
    ]
    conjugate_feat = [1.0 if bond.GetIsConjugated() else 0.0]
    ring_feat = [1.0 if bond.IsInRing() else 0.0]
    return np.array(bond_type_feat + conjugate_feat + ring_feat, dtype=np.float32)

def calc_chemical_affinity(drug_smiles, food_smiles):
    key_groups = {
        "hydroxyl": "[OX2H]",
        "carboxyl": "[$([CX3](=O)[OX2H])]",
        "amino": "[NX3;H2,H1;!$(NC=O)]",
        "carbonyl": "[$([CX3]=O)]",
        "aromatic": "[c]",
        "halogen": "[F,Cl,Br,I]"
    }

    def count_groups(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return np.zeros(len(key_groups))
        vec = []
        for smarts in key_groups.values():
            patt = Chem.MolFromSmarts(smarts)
            cnt = len(mol.GetSubstructMatches(patt)) if patt else 0
            vec.append(cnt)
        return np.array(vec, dtype=np.float32)

    d_vec = count_groups(drug_smiles)
    f_vec = count_groups(food_smiles)
    norm_d = np.linalg.norm(d_vec)
    norm_f = np.linalg.norm(f_vec)
    if norm_d < 1e-6 or norm_f < 1e-6:
        sim = 0.1
    else:
        sim = np.dot(d_vec, f_vec) / (norm_d * norm_f)
    return float(np.clip(sim, 0.01, 0.99))


class GATEncoder(nn.Module):
    def __init__(self, input_dim=55, edge_dim=6, hidden_dim=256, heads=4):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True, edge_dim=edge_dim, dropout=0.1)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, edge_dim=edge_dim, dropout=0.1)
        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        x1 = self.conv1(x, edge_index, edge_attr)
        x1 = self.norm1(x1)
        x1 = F.gelu(x1)
        x2 = self.conv2(x1, edge_index, edge_attr)
        x2 = self.norm2(x2)
        return x2


class GraphExtractPart(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_gat = GATEncoder()
        # 双池512维投影压缩至256，适配酶增强模块输入
        self.pool_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )

    def build_mol_graph(self, smi, atom_feat_dict):
        mol = Chem.MolFromSmiles(smi)
        atom_f = atom_feat_dict.get("atom_fea", None)
        if mol and atom_f is not None and atom_f.shape[0] == mol.GetNumAtoms():
            x = atom_f.clone().detach().float()
            edges = []
            bf_list = []
            for b in mol.GetBonds():
                u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                edges.append([u, v])
                edges.append([v, u])
                bf = get_bond_feature(b)
                bf_list.append(bf)
                bf_list.append(bf)
            ei = torch.LongTensor(edges).t()
            if bf_list:
                ea_np = np.stack(bf_list, axis=0)
                ea = torch.from_numpy(ea_np)
            else:
                ea = torch.empty((0, 6), dtype=torch.float32)
            return Data(x=x, edge_index=ei, edge_attr=ea)
        else:
            return Data(
                x=torch.zeros(1, 55, dtype=torch.float32),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty((0, 6), dtype=torch.float32)
            )

    def pack(self, data_batch):
        drug_ids = data_batch["drug_id"]
        food_ids = data_batch["food_id"]
        drug_graphs = []
        food_graphs = []
        aff_list = []
        for did, fid in zip(drug_ids, food_ids):
            d_graph = self.build_mol_graph(drug_smiles.get(did, ""), drug_atom_fea.get(did, {}))
            f_graph = self.build_mol_graph(food_smiles.get(fid, ""), food_atom_fea.get(fid, {}))
            drug_graphs.append(d_graph)
            food_graphs.append(f_graph)
            aff = calc_chemical_affinity(drug_smiles.get(did, ""), food_smiles.get(fid, ""))
            aff_list.append(aff)
        aff_tensor = torch.tensor(aff_list, dtype=torch.float32).unsqueeze(1).unsqueeze(2)
        return drug_graphs, food_graphs, aff_tensor

    def forward(self, batch_data):
        drug_graphs, food_graphs, affinity = self.pack(batch_data)
        db = Batch.from_data_list(drug_graphs).to(device)
        fb = Batch.from_data_list(food_graphs).to(device)

        drug_node = self.shared_gat(db.x, db.edge_index, db.edge_attr)
        food_node = self.shared_gat(fb.x, fb.edge_index, fb.edge_attr)

        drug_mean = global_mean_pool(drug_node, db.batch)
        drug_max = global_max_pool(drug_node, db.batch)
        drug_pool_cat = torch.cat([drug_mean, drug_max], dim=-1)  # [B,512]
        drug_pool = self.pool_proj(drug_pool_cat).unsqueeze(1)   # [B,1,256]

        food_mean = global_mean_pool(food_node, fb.batch)
        food_max = global_max_pool(food_node, fb.batch)
        food_pool_cat = torch.cat([food_mean, food_max], dim=-1) # [B,512]
        food_pool = self.pool_proj(food_pool_cat).unsqueeze(1)    # [B,1,256]

        affinity = affinity.to(device)
        return drug_pool, food_pool, affinity


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.cross_d2f = ImprovedMultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout)
        self.cross_f2d = ImprovedMultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout)
        self.norm = nn.LayerNorm(dim * 2)
        self.gate = nn.Sequential(nn.Linear(dim*2, dim*2), nn.GELU(), nn.Dropout(0.1))

    def forward(self, drug_emb, food_emb):
        # 重点修复：去掉 query= key= value= 关键字参数
        d2f = self.cross_d2f(drug_emb, food_emb, food_emb)
        f2d = self.cross_f2d(food_emb, drug_emb, drug_emb)
        fuse_raw = torch.cat([d2f, f2d], dim=-1)
        gate_weight = self.gate(fuse_raw)
        fuse_out = self.norm(fuse_raw + gate_weight)
        return fuse_out


class DFIFusionModel(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4):
        super().__init__()
        self.graph_extractor = GraphExtractPart()

        raw_cyp = torch.load("cyp450_embed.pt", map_location=device)
        self.register_buffer("cyp450_emb", raw_cyp)

        self.drug_enhance = EnzymeGuidedEnhance(hidden_dim=embed_dim, num_enzyme=5, num_heads=num_heads, dropout=0.1)
        self.food_enhance = EnzymeGuidedEnhance(hidden_dim=embed_dim, num_enzyme=5, num_heads=num_heads, dropout=0.1)

        self.aff_gate = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.Sigmoid()
        )
        self.cross_fusion = CrossAttentionFusion(dim=embed_dim, heads=num_heads)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim*2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.18),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.18),
            nn.Linear(256, 2)
        )
        self.apply(self._init_weight)

    def _init_weight(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, batch_data):
        drug_feat, food_feat, affinity = self.graph_extractor(batch_data)

        drug_feat, _ = self.drug_enhance(drug_feat, self.cyp450_emb)
        food_feat, _ = self.food_enhance(food_feat, self.cyp450_emb)

        gate_weight = self.aff_gate(affinity)
        drug_feat = drug_feat * gate_weight
        food_feat = food_feat * gate_weight

        fuse = self.cross_fusion(drug_feat, food_feat)
        fuse = fuse.squeeze(1)
        logits = self.mlp(fuse)
        return logits