import torch
import torch.nn as nn
import math

class RoPE(nn.Module):
    '''positional encoding with sin and cos

    Args:
        dim (int): length of the embedding vector, same as embedding dim
        max_len (int): max length of the input sequence
    '''
    def __init__(self, dim, max_len=5000):
        super(RoPE, self).__init__()
 
        # positional encoding matrix
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)   
        pe[:, 1::2] = torch.cos(position * div_term)   
        # batch size occupies one dim
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    def forward(self, x):
        # x: [batch_size, seq_len, dim] embedding
        x = x + self.pe[:, :x.size(1)].clone().detach() 
        return x
    
    