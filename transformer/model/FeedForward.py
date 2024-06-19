import torch
import torch.nn as nn
from .Norm import LayerNorm

class FeedForward(nn.Module):
    '''position-wise feed-forward network

    Args:
        dim (int): dimension of input/output
        hidden_dim (int): dimension of hidden layer
        dropout (float): dropout rate
    '''
    def __init__(self, dim = 512, hidden_dim = 2048):
        super(FeedForward, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )
        self.layer_norm = LayerNorm(dim, eps=1e-6)
        
    def forward(self, x):
        residual = x
        x = self.linear(x)
        # add in add & norm
        x += residual
        # norm in add &norm
        x = self.layer_norm(x)
        return x