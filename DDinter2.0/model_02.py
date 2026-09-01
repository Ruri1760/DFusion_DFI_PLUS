import torch
import torch.nn as nn
import pickle
from atom_singal_block import DFIFusionModel as model_two
from txt_singal_block import fusion_model as model_one

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def open_files(file_name):
    with open("./pre_params/"+file_name, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

drug_food_ID = open_files("drug_food_ID.pkl")
ID_NUM = len(drug_food_ID)



# 1. 文本+结构 主干融合（保留原有最优模块）
class BilayerGatedFusion(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.dim = dim
        self.txt_avg_proj = nn.Linear(dim, dim)
        self.txt_max_proj = nn.Linear(dim, dim)
        self.struc_proj = nn.Linear(dim, dim)

        self.modal_gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.Sigmoid()
        )
        self.feat_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Sigmoid()
        )
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(0.15)

    def forward(self, txt_feat, struc_feat):
        # txt_feat: [B, seq_len, 512]
        # struc_feat: [B,1,512]
        txt_avg = txt_feat.mean(dim=1)
        txt_max = txt_feat.max(dim=1)[0]
        struc = struc_feat.squeeze(1) # [B,512]

        t_avg = self.txt_avg_proj(txt_avg)
        t_max = self.txt_max_proj(txt_max)
        s = self.struc_proj(struc)

        concat = torch.cat([t_avg, t_max, s], dim=-1)
        m_gate = self.modal_gate(concat)
        fuse = t_avg * m_gate + t_max * (1 - m_gate) + s
        f_gate = self.feat_gate(fuse)
        fuse = fuse * f_gate
        fuse = self.norm(self.out_proj(fuse) + fuse)
        return self.drop(fuse)


# 2. 双ID交互模块（核心：drug_id + food_id 配对融合）
class PairIDFusion(nn.Module):
    def __init__(self, emb_dim, out_dim=512):
        super().__init__()
        self.interact = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU()
        )
        self.id_proj = nn.Linear(emb_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, drug_emb, food_emb):
        id_cat = torch.cat([drug_emb, food_emb], dim=-1)
        id_inter = self.interact(id_cat)
        id_feat = self.id_proj(id_inter)
        return self.norm(id_feat)


# 3. 模态特征 + ID配对特征 融合
class ModalIDJointFusion(nn.Module):
    def __init__(self, feat_dim=512, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.feat_proj = nn.Linear(feat_dim, feat_dim)
        self.joint_gate = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.Sigmoid()
        )
        self.final_enhance = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Dropout(0.15)
        )

    def forward(self, modal_feat, pair_id_feat):
        modal_feat = self.feat_proj(modal_feat)
        concat = torch.cat([modal_feat, pair_id_feat], dim=-1)
        gate = self.joint_gate(concat)
        fused = self.alpha * modal_feat * gate + (1 - self.alpha) * pair_id_feat * (1 - gate)
        return self.final_enhance(fused)


# 4. 分类头
class AUCOptClassifier(nn.Module):
    def __init__(self, dim=512, num_classes=2):
        super().__init__()
        self.cls = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.LayerNorm(dim//2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim//2, num_classes)
        )
    def forward(self, x):
        return self.cls(x)


# 主模型
class fusion_model(nn.Module):
    def __init__(self, flag=1, hidden1=256):
        super().__init__()
        self.hidden1 = hidden1
        id_emb_dim = hidden1 * 2
        self.struct_model = model_two().to(device)
        self.txt_model = model_one().to(device)

        self.ID_embedding = nn.Embedding(ID_NUM, id_emb_dim)

        self.modal_fusion = BilayerGatedFusion(dim=512)
        self.pair_id_fusion = PairIDFusion(emb_dim=id_emb_dim, out_dim=512)
        self.joint_fusion = ModalIDJointFusion(feat_dim=512, alpha=0.7)
        self.classifier = AUCOptClassifier(dim=512, num_classes=4)

    def forward(self, dataset_batch):
        # dataset_batch = ((drug_list, food_list), labels)  tuple输入，和dataloader输出对齐
        data_tuple, _ = dataset_batch
        drug_list, food_list = data_tuple
        N = len(drug_list)

   
        feat_struct = self.struct_model(dataset_batch)   # [B,1,512]
        feat_txt = self.txt_model(dataset_batch)         # [B,seq_len,512]

        # -------- 2. drug_id food_id embedding --------
        drug_IDs, food_IDs = [], []
        for i in range(N):
            drug_IDs.append(drug_food_ID[drug_list[i]])
            food_IDs.append(drug_food_ID[food_list[i]])
        drug_IDs = torch.tensor(drug_IDs, dtype=torch.int).to(device)
        food_IDs = torch.tensor(food_IDs, dtype=torch.int).to(device)

        drug_emb = self.ID_embedding(drug_IDs)
        food_emb = self.ID_embedding(food_IDs)

        # --------3. ID配对交互特征 --------
        pair_id_feat = self.pair_id_fusion(drug_emb, food_emb)

        # --------4. 文本+结构门控融合 --------
        modal_feat = self.modal_fusion(feat_txt, feat_struct)

        # --------5. 模态特征与ID特征联合融合 --------
        total_feat = self.joint_fusion(modal_feat, pair_id_feat)

        # --------6.输出logits [B,2] --------
        logits = self.classifier(total_feat)
        return logits