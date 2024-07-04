import torch
import torch.nn as nn
from .Attention import MultiAttention
from .FeedForward import FeedForward
from .Embedding import Embeddings
from .Encoding import RoPE
from .Mask import mask_subsequence

# for decoder architecture
class DecoderLayer(nn.Module):
    '''one layer of decoder
    
    Args:
        dim (int): length of the embedding vector
        ff_dim (int): length of the hidden layer in the feedforward network
        n_heads (int): number of heads
    '''
    def __init__(self, dim, ff_dim, n_heads):
        super(DecoderLayer, self).__init__()
        self.decoder_masked_atten = MultiAttention(dim=dim, n_heads=n_heads, dim_k=dim//n_heads, dim_v=dim//n_heads)
        self.ffnet = FeedForward(dim=dim, hidden_dim=ff_dim)
    
    # inputs is the input of the decoder
    # masked_atten_mask is the mask for the masked multi-head attention
    def forward(self, inputs, masked_atten_mask):
        outputs, masked_attention = self.decoder_masked_atten(inputs, inputs, inputs, masked_atten_mask)
        # the results from last masked multi-head attention is used as Q
        outputs = self.ffnet(outputs)
        return outputs, masked_attention
    
# for encoder-decoder architecture
class Decoder(nn.Module):
    '''decoder itself

    Args:
        vocab (int): number of words in the vocabulary
        emb_dim (int): length of the embedding vector
        ff_dim (int): length of the hidden layer in the feedforward network
        context_len (int): length of the context
        n_layers (int): number of layers
        n_heads (int): number of heads
        device
    '''
    def __init__(self, vocab, emb_dim, ff_dim, context_len, n_layers = 6, n_heads = 8, device = 'cuda'):
        super(Decoder, self).__init__()
        self.embedding = Embeddings(emb_dim, vocab)
        self.encoding = RoPE(emb_dim)
        self.layers = nn.ModuleList([DecoderLayer(emb_dim, ff_dim, n_heads) for _ in range(n_layers)])
        self.device = device
       
    # inputs is the input of the decoder
    def forward(self, inputs): 
        # mask out stop words and subsequence in the inputs
        masked_atten_mask = mask_subsequence(inputs).to(self.device)
        # embedding & encoding
        embedding = self.embedding(inputs)
        outputs = self.encoding(embedding)
        # decode
        for layer in self.layers:
            outputs, _ = layer(outputs, masked_atten_mask)
        return outputs

        