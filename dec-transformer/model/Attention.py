import torch
import torch.nn as nn
from .Norm import LayerNorm

class SelfAttention(nn.Module):
    '''self attention by dot-product

    Args:
        scale_factor (): scale_factor
        dropout (float): dropout rate
    '''
    def __init__(self, scale_factor, dropout=0.0):
        super(SelfAttention, self).__init__()
        self.scale_factor = scale_factor
        # self.dropout = nn.Dropout(dropout)
 
    # (4d_k + 3)len*len (should multiply by heads)
    def forward(self, q, k, v, mask=None):
        # matmul & scale
        # 2*d_k * len * len + len*len
        scores = torch.matmul(q, k.transpose(2, 3)) / self.scale_factor
 
        # optional mask
        # 2*len*len
        if mask is not None:
            # use -1e9 as the large negative value to mask the padding tokens
            scores = scores.masked_fill(mask == 0, -1e9)
        # softmax
        # 3 * len * len
        scores = torch.softmax(scores, dim=-1)
        # matmul
        # 2*d_k * len * len
        output = torch.matmul(scores, v)
        # 返回 output和注意力分数
        return output, scores
    

class MultiAttention(nn.Module):
    '''multi-head attention

    Args:
        n_heads (int): number of heads
        dim (int): length of the embedding vector
        dim_k (int): dim of k
        dim_v (int): dim of v
        dropout (float): dropout rate
    '''
    def __init__(self, n_heads = 8, dim = 512, dim_k = 64, dim_v = 64,  dropout=0.1):
        super(MultiAttention, self).__init__()
        self.n_heads = n_heads
        self.dim_k = dim_k
        self.dim_v = dim_v
 
        # weight matrices for Q, K, V
        # the linear layer represents the weight matrices
        self.Wq = nn.Linear(dim, n_heads * dim_k, bias=False)
        self.Wk = nn.Linear(dim, n_heads * dim_k, bias=False)
        self.Wv = nn.Linear(dim, n_heads * dim_v, bias=False)
        self.fc = nn.Linear(n_heads * dim_v, dim, bias=False)
 
        self.attention = SelfAttention(scale_factor=dim_k ** 0.5)
 
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(dim, eps=1e-6)
 
    # 8 * len * d_model * d_model + 4 * len * len * d_model + 3 * len * len * h + 9 * len * d_model
    def forward(self, q, k, v, mask=None):
        # q, k, v：[batch_size, seq_num, dim]
        # len_k为输入的序列长度
        batch_size = q.size(0)
        # for residual connection
        residual = q
        # breakpoint()
 
        # multiplied by W^Q, W^K, W^V
        # (batch_size, length, n_heads, dim_k) => (batch_size, n_heads, length, dim_k)
        # 3 * 2 * len * d_model * d_model
        query = self.Wq(q).view(batch_size, -1, self.n_heads, self.dim_k).transpose(1, 2)
        key   = self.Wk(k).view(batch_size, -1, self.n_heads, self.dim_k).transpose(1, 2)
        value = self.Wv(v).view(batch_size, -1, self.n_heads, self.dim_v).transpose(1, 2)
 
        if mask is not None:
            mask = mask.unsqueeze(1) 
        
        # (4d_k + 3) * len * len * h => 
        # 4 * d_model * len * len + 3 * len * len * h
        x, attn = self.attention(query, key, value, mask=mask)
 
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        # (batch_size, n_heads, length, d_k/d_v) => (batch_size, length, n_heads, d_k/d_v) => (batch_size, length, dim)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.dim_k)
        # the final linear layer
        # 2 * len * d_model * d_model
        # > we apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized.
        x = self.dropout(self.fc(x))
        # add in add & norm
        # len * d_model
        x += residual
        # norm in add & norm
        # 8 * len * d_model
        x = self.layer_norm(x)
        return x, attn
    