import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class MultiheadAttention(nn.Module):
    def __init__(self, n_head, hidden_dim, drop_out=0.05):
        super(MultiheadAttention, self).__init__()
        self.n_head = n_head
        self.dim = hidden_dim
        self.d_k = self.dim // self.n_head  

        self.wq = nn.Linear(self.dim, self.dim)
        self.wk = nn.Linear(self.dim, self.dim)
        self.wv = nn.Linear(self.dim, self.dim)

        self.softmax = nn.Softmax(dim=-1) 
        self.f = nn.Linear(self.dim, self.dim)
        self.norm = nn.LayerNorm(self.dim)
        self.dropout = nn.Dropout(drop_out)

    def split(self, tensor):

        batch_size, seq_len, hidden_dim = tensor.size()
        return tensor.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)

    def concat(self, tensor):
        batch_size, n_head, seq_len, d_k = tensor.size()

        return tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)

    def attention(self, k, q, v, mask=None):

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  

        attn_weight = self.softmax(scores)  
        output = torch.matmul(attn_weight, v)
        return output, scores

    def forward(self, x, y, z, mask=None):
        q, k, v = self.wq(x), self.wk(y), self.wv(z) 
        q, k, v = self.split(q), self.split(k), self.split(v) 
        
        # 计算注意力
        output, attn_weight = self.attention(k, q, v, mask) 
        output = self.concat(output)  
        output = self.f(output)  
        output = self.dropout(output)
        
        output = self.norm(output + x)
        return output, attn_weight

class Attention(nn.Module):
    def __init__(self, hidden_dim, drop_out=0.05):
        super(Attention, self).__init__()
        self.dim = hidden_dim

        self.wq = nn.Linear(self.dim, self.dim)
        self.wk = nn.Linear(self.dim, self.dim)
        self.wv = nn.Linear(self.dim, self.dim)

        self.softmax = nn.Softmax(dim=-1) 
        self.f = nn.Linear(self.dim, self.dim)
        self.norm = nn.LayerNorm(self.dim)
        self.dropout = nn.Dropout(drop_out)
        nn.init.xavier_uniform_(self.wq.weight)  # Xavier确保输入输出方差一致
        nn.init.xavier_uniform_(self.wk.weight)
        nn.init.xavier_uniform_(self.wv.weight)
        nn.init.xavier_uniform_(self.f.weight)
        # 偏置初始化（0.1兜底，避免输出全零）
        nn.init.constant_(self.wq.bias, 0.1)
        nn.init.constant_(self.wk.bias, 0.1)
        nn.init.constant_(self.wv.bias, 0.1)
        nn.init.constant_(self.f.bias, 0.1)
    def attention(self, k, q, v):

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dim)

        attn_weight = self.softmax(scores)  
        output = torch.matmul(attn_weight, v)
        return output, scores

    def forward(self, x):
        q, k, v = self.wq(x), self.wk(x), self.wv(x) 
 
        # 计算注意力
        output, attn_weight = self.attention(k, q, v) 
        output = self.f(output)  
        output = self.dropout(output)
        
        output = self.norm(output + x)
        return output, attn_weight
    
class ImprovedMultiheadAttention(nn.Module):
    def __init__(self, num_heads=4, embed_dim=256, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "It is mandatory that embed_dim is an integer multiple of num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)  
        
        nn.init.xavier_uniform_(self.q_proj.weight) 
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        nn.init.constant_(self.q_proj.bias, 0.1)
        nn.init.constant_(self.k_proj.bias, 0.1)
        nn.init.constant_(self.v_proj.bias, 0.1)
        nn.init.constant_(self.out_proj.bias, 0.1)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim) 

    def forward(self, q, k, v, key_padding_mask=None):
        batch_size = q.size(0)
        
        q = self.q_proj(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1))  
        attn_scores = attn_scores / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32, device=device))
        
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(1).to(device)
            attn_scores = attn_scores.masked_fill(mask, -1e9)  
        
        attn_weights = F.softmax(attn_scores, dim=-1) 
        attn_weights = self.dropout(attn_weights)  
        
        attn_output = torch.matmul(attn_weights, v)  
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        attn_output = self.out_proj(attn_output) 
        
        output = self.norm(attn_output + q.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim))
        
        return output