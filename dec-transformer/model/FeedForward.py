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
    def __init__(self, dim = 512, hidden_dim = 2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.ReLU(),
            # > we apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized.
            nn.Linear(hidden_dim, dim, bias=False),
            nn.Dropout(dropout),
        )
        self.layer_norm = LayerNorm(dim, eps=1e-6)
        
    # 4 * len*d_model * d_ff + 9 * len * d_model
    def forward(self, x):
        # x: [batch_size, len, dim]
        residual = x
        x = self.linear(x)
        # add in add & norm
        x += residual
        # norm in add &norm
        x = self.layer_norm(x)
        return x