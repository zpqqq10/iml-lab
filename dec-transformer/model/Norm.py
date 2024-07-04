import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    '''layer normalization

    Args:
        shape (int): length of the embedding vector
        eps (float): a small number to prevent division by zero
    '''
    def __init__(self, shape, eps = 1e-5):
        super(LayerNorm, self).__init__()
        # initialize two learnable parameters
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # prevent division by zero
        self.eps = eps
        
    # 8 * len * d_model
    def forward(self, x):
        # mean and variance
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased = False, keepdim=True)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * normalized + self.beta
 