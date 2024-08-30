from math import sqrt
import torch
import torch.nn.functional as F
from torch import nn

def scaled_dot_product_attention(query, key, value, mask):
    dim_k = query.size(-1) # size of hidden state
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    scores = scores.masked_fill(mask == 0, -1e9) # fill 0 mask with super small number so it wont affect the softmax weight
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim) 
        self.k = nn.Linear(embed_dim, head_dim) 
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_state, mask):
        attn_outputs = scaled_dot_product_attention(
        self.q(hidden_state), self.k(hidden_state), self.v(hidden_state), mask) 
        return attn_outputs

class MultiHeadAttention(nn.Module): 
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        head_dim = embed_dim // num_heads 
        self.heads = nn.ModuleList([AttentionHead(embed_dim, head_dim) for _ in range(num_heads)])
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state, mask):
        x = torch.cat([h(hidden_state, mask) for h in self.heads], dim=-1) 
        x = self.output_linear(x)
        return x

class FeedForward(nn.Module): 
    def __init__(self, hidden_size, intermediate_size=3072, hidden_dropout_prob=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(hidden_size, intermediate_size) 
        self.linear_2 = nn.Linear(intermediate_size, hidden_size) 
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x