import torch
import torch.nn as nn
import pickle

def open_files(file_name):
    with open("./pre_params/"+file_name, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

drug_dict_sw = open_files("drug_fea.pkl")
food_dict_sw = open_files("food_fea.pkl")
drug_food_ID = open_files("drug_food_ID.pkl")
cypi_dict = open_files("cypi_dict.pkl")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==================== 你原始 Encoder 完全不变 ====================
class EncoderLayer(nn.Module):
    def __init__(self, i_channel, o_channel, growth_rate, groups, pad2=7):
        super(EncoderLayer, self).__init__()
        self.conv1 = nn.Conv1d(i_channel, o_channel, 2*pad2+1, 1, pad2, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm1d(i_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(o_channel, growth_rate, 2*pad2+1, 1, pad2, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm1d(o_channel)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        xn = self.bn1(x)
        xn = self.relu(xn)
        xn = self.conv1(xn)
        xn = self.bn2(xn)
        xn = self.relu(xn)
        xn = self.conv2(xn)
        xn = self.drop(xn)
        return torch.cat([x, xn], 1)

class Encoder(nn.Module):
    def __init__(self, inc, outc, growth_rate, layers, groups, pad1=7, pad2=3):
        super(Encoder, self).__init__()
        self.layers = layers
        self.relu = nn.ReLU(inplace=True)
        self.conv_in = nn.Conv1d(inc, inc, 2*pad1+1, 1, pad1, bias=False)
        self.dense_cnn = nn.ModuleList([
            EncoderLayer(inc + growth_rate * i, inc + (growth_rate//2)*i, growth_rate, groups, pad2)
            for i in range(layers)
        ])
        self.conv_out = nn.Conv1d(inc + growth_rate*layers, outc, 2*pad1+1, 1, pad1, bias=False)

    def forward(self, x):
        x = self.conv_in(x)
        for i in range(self.layers):
            x = self.dense_cnn[i](x)
        x = self.relu(x)
        x = self.conv_out(x)
        x = self.relu(x)
        return x



# ==================== 主模型：完全保留你原始结构 ====================
class DFusion_DFI_model(nn.Module):
    def __init__(self, hidden1=256, attn_heads=1):
        super().__init__()
        self.hidden1 = hidden1

        # CNN编码器
        self.encoder_DF = Encoder(768, hidden1, 128, 3, groups=32, pad1=7, pad2=3)

        # ==================== CYP 交叉注意力模块 ====================
        self.cyp_proj = nn.Linear(1024, 768)

        # 注意力层
        self.drug_multihead = nn.TransformerEncoderLayer(hidden1, 1, hidden1*2, 0.05, batch_first=True)
        self.food_multihead = nn.TransformerEncoderLayer(hidden1, 1, hidden1*2, 0.05, batch_first=True)
        self.DF_Multihead = nn.TransformerEncoderLayer(hidden1*2, 1, hidden1*6, 0.05, batch_first=True)

        # 你原始的池化、全连接
        self.avg_pool = nn.ModuleList([
            nn.AdaptiveAvgPool1d(128),
            nn.AdaptiveAvgPool1d(128)
        ])
        self.drop_out = nn.ModuleList([nn.Dropout(0.1), nn.Dropout(0.05)])
        self.DFI_feature = nn.ModuleList([nn.Linear(hidden1*2, 512), nn.Linear(512, 128)])
        self.Batch = nn.ModuleList([nn.LayerNorm(512), nn.LayerNorm(128)])
        self.act = nn.GELU()
        self.DFI_Pre = nn.Linear(128, 2)

    def get_cyp_sequence(self, drug_list):
        B = len(drug_list)
        cyp_raw = torch.zeros(B, 3, 1024).to(device)
        for i, d in enumerate(drug_list):
            if d in cypi_dict:
                cyp_raw[i] = torch.tensor(cypi_dict[d]).float().to(device)
        # 映射到 768 维
        cyp_seq = self.cyp_proj(cyp_raw)
        return cyp_seq

    def pack(self, data_batch):
        drug_list, food_list = data_batch['drug_id'], data_batch["food_id"]
        N, max_len = len(drug_list), 100
        drug_fea = torch.zeros(N, max_len, 768).to(device)
        food_fea = torch.zeros(N, max_len, 768).to(device)

        for i in range(N):
            if food_list[i] in food_dict_sw:
                t = food_dict_sw[food_list[i]]
                L = min(t.shape[0], max_len)
                food_fea[i, :L] = torch.tensor(t[:L]).to(device)
            if drug_list[i] in drug_dict_sw:
                t = drug_dict_sw[drug_list[i]]
                L = min(t.shape[0], max_len)
                drug_fea[i, :L] = torch.tensor(t[:L]).to(device)

        return drug_fea, food_fea, drug_list

    def forward(self, dataset_batch):
        drug_fea, food_fea, drug_list = self.pack(dataset_batch)


        drug_fea = self.encoder_DF(drug_fea.permute(0,2,1)).permute(0,2,1)
        food_fea = self.encoder_DF(food_fea.permute(0,2,1)).permute(0,2,1)

        # 自注意力
        drug_fea = self.drug_multihead(drug_fea)
        food_fea = self.food_multihead(food_fea)

        # 你原始的融合逻辑
        temp = drug_fea + food_fea
        drug_fea = self.avg_pool[0](drug_fea)
        food_fea = self.avg_pool[1](food_fea)

        DF_Feature = torch.cat([drug_fea, food_fea, temp], dim=-1)
        DF_Feature = self.DF_Multihead(DF_Feature)
 
        return DF_Feature