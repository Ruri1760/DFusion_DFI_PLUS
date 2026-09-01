import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F



class cross_attention(nn.Module):
    def __init__(self, hidden1, hidden2):
        super(cross_attention, self).__init__()
        self.W_q = nn.Linear(hidden1, hidden2)
        self.W_k = nn.Linear(hidden2, hidden2)
        self.W_v = nn.Linear(hidden2, hidden2)
        self.drop = nn.Dropout(p=0.1)
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(-1)

    def forward(self, x, y, z): # xs, x两个模态的表示，先做交叠注意力，然后再拼接
        q = self.W_q(x)
        k = self.W_k(y)
        v = self.W_v(z)
        weight = torch.matmul(q.permute(0, 2, 1), k)
        scale = weight.size(-1) ** -0.5
        weights = self.softmax(weight * scale)
        ys = torch.matmul(v,self.drop(weights)) + x+y
        return ys


