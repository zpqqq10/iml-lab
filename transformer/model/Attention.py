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
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, q, k, v, mask=None):
        # matmul & scale
        scores = torch.matmul(q, k.transpose(2, 3)) / self.scale_factor
 
        # optional mask
        if mask is not None:
            # use -1e9 as the large negative value to mask the padding tokens
            scores = scores.masked_fill(mask == 0, -1e9)
        # softmax
        scores = self.dropout(torch.softmax(scores, dim=-1))
        # matmul
        output = torch.matmul(scores, v)
        # 返回 output和注意力分数
        return output, scores
    

class MultiAttention(nn.Module):
    '''multi-head attention

    Args:
        n_heads (int): number of heads
        dim (int): length of the embedding vector
        dim_k (int): dim of k
        dropout (float): dropout rate
    '''
    def __init__(self, n_heads = 8, dim = 512, dim_k = 64, dropout=0.1):
        super(MultiAttention, self).__init__()
        self.n_heads = n_heads
        self.dim_k = dim_k
 
        # weight matrices for Q, K, V
        # the linear layer represents the weight matrices
        self.Wq = nn.Linear(dim, n_heads * dim_k, bias=False)
        self.Wk = nn.Linear(dim, n_heads * dim_k, bias=False)
        self.Wv = nn.Linear(dim, n_heads * dim_k, bias=False)
        self.fc = nn.Linear(n_heads * dim_k, dim, bias=False)
 
        self.attention = SelfAttention(scale_factor=dim_k ** 0.5)
 
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(dim, eps=1e-6)
 
    def forward(self, q, k, v, mask=None):
        # q, k, v：[batch_size, seq_num, dim]
        # len_k为输入的序列长度
        batch_size = q.size(0)
        # for residual connection
        residual = q
        # breakpoint()
 
        # multiplied by W^Q, W^K, W^V
        # (batch_size, length, n_heads, dim_k) => (batch_size, n_heads, length, dim_k)
        query = self.Wq(q).view(batch_size, -1, self.n_heads, self.dim_k).transpose(1, 2)
        key   = self.Wk(k).view(batch_size, -1, self.n_heads, self.dim_k).transpose(1, 2)
        value = self.Wv(v).view(batch_size, -1, self.n_heads, self.dim_k).transpose(1, 2)
 
        if mask is not None:
            mask = mask.unsqueeze(1) 
        
        x, attn = self.attention(query, key, value, mask=mask)
 
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        # (batch_size, 8, len_k, 64) => (batch_size, len_k, 8, 64) => (batch_size, len_k, 512)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.dim_k)
        # the final linear layer
        x = self.fc(x)
        # add in add & norm
        x += residual
        # norm in add & norm
        x = self.layer_norm(x)
        return self.dropout(x), attn
    