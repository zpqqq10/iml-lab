import torch
import torch.nn as nn
import math

class Embeddings(nn.Module):
    '''word embeddings

    Args:
        dim (int): length of the embedding vector
        vocab (int): number of words in the vocabulary
    '''
    def __init__(self, dim, vocab):
        super(Embeddings, self).__init__()
        # 调用nn.Embedding预定义层，获得实例化词嵌入对象self.lut
        self.lut = nn.Embedding(vocab, dim)
        self.dim = dim  #表示词向量维度
 
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.dim)