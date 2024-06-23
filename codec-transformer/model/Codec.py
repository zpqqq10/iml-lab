import torch
import torch.nn as nn
from .Attention import MultiAttention
from .FeedForward import FeedForward
from .Embedding import Embeddings
from .Encoding import RoPE
from .Mask import mask_stop_words, mask_subsequence

class EncoderLayer(nn.Module):
    '''one layer of encoder
    
    Args:
        dim (int): length of the embedding vector
        ff_dim (int): length of the hidden layer in the feedforward network
        n_heads (int): number of heads
    '''
    def __init__(self, dim, ff_dim, n_heads):
        super(EncoderLayer, self).__init__()
        self.encoder_atten = MultiAttention(dim=dim, n_heads=n_heads, dim_k=dim//n_heads, dim_v=dim//n_heads)
        self.ffnet = FeedForward(dim=dim, hidden_dim=ff_dim)
        
    def forward(self, inputs, mask):
        # the same inputs for q, k, v
        outputs, attention = self.encoder_atten(inputs, inputs, inputs, mask)
        outputs = self.ffnet(outputs)
        return outputs, attention
    
class Encoder(nn.Module):
    '''encoder itself

    Args:
        vocab (int): number of words in the vocabulary
        emb_dim (int): length of the embedding vector
        ff_dim (int): length of the hidden layer in the feedforward network
        n_layers (int): number of layers
        n_heads (int): number of heads
        device
    '''
    def __init__(self, vocab, emb_dim, ff_dim , n_layers = 6, n_heads = 8, device = 'cuda'):
        super(Encoder, self).__init__()
        self.embedding = Embeddings(emb_dim, vocab)
        self.encoding = RoPE(emb_dim)
        self.layers = nn.ModuleList([EncoderLayer(emb_dim, ff_dim, n_heads) for _ in range(n_layers)])
        self.device = device
        
    def forward(self, inputs): 
        # mask out stop words in the inputs
        mask = mask_stop_words(inputs, inputs)
        # embedding & encoding
        embedding = self.embedding(inputs)
        outputs = self.encoding(embedding)
        # encode
        attentions = []
        for layer in self.layers:
            outputs, attention = layer(outputs, mask)
            attentions.append(attention)
        return outputs, attentions
    
# for encoder-decoder architecture
class DecoderLayer(nn.Module):
    '''one layer of decoder
    
    Args:
        dim (int): length of the embedding vector
        ff_dim (int): length of the hidden layer in the feedforward network
        n_heads (int): number of heads
    '''
    def __init__(self, dim, ff_dim, n_heads):
        super(DecoderLayer, self).__init__()
        self.decoder_atten = MultiAttention(dim=dim, n_heads=n_heads, dim_k=dim//n_heads, dim_v=dim//n_heads)
        self.decoder_masked_atten = MultiAttention(dim=dim, n_heads=n_heads, dim_k=dim//n_heads, dim_v=dim//n_heads)
        self.ffnet = FeedForward(dim=dim, hidden_dim=ff_dim)
    
    # inputs is the input of the decoder
    # enc_outputs is the output of the encoder
    # masked_atten_mask is the mask for the masked multi-head attention
    # enc_atten_mask is the mask for the encoder-decoder multi-head attention
    def forward(self, inputs, enc_outputs, masked_atten_mask, enc_atten_mask):
        outputs, masked_attention = self.decoder_masked_atten(inputs, inputs, inputs, masked_atten_mask)
        # the results from last masked multi-head attention is used as Q
        outputs, attention = self.decoder_atten(outputs, enc_outputs, enc_outputs, enc_atten_mask)
        outputs = self.ffnet(outputs)
        return outputs, masked_attention, attention
    
# for encoder-decoder architecture
class Decoder(nn.Module):
    '''decoder itself

    Args:
        vocab (int): number of words in the vocabulary
        emb_dim (int): length of the embedding vector
        ff_dim (int): length of the hidden layer in the feedforward network
        n_layers (int): number of layers
        n_heads (int): number of heads
        device
    '''
    def __init__(self, vocab, emb_dim, ff_dim,  n_layers = 6, n_heads = 8, device = 'cuda'):
        super(Decoder, self).__init__()
        self.embedding = Embeddings(emb_dim, vocab)
        self.encoding = RoPE(emb_dim)
        self.layers = nn.ModuleList([DecoderLayer(emb_dim, ff_dim, n_heads) for _ in range(n_layers)])
        self.device = device
       
    # inputs is the input of the decoder
    # enc_inputs is the input of the encoder 
    # enc_outputs is the output of the encoder 
    def forward(self, inputs, enc_inputs, enc_outputs): 
        # mask out stop words and subsequence in the inputs
        masked_atten_mask = mask_subsequence(inputs).to(self.device)
        masked_atten_mask = masked_atten_mask.bool() & mask_stop_words(inputs, inputs).to(self.device)
        enc_atten_mask = mask_stop_words(enc_inputs, inputs).to(self.device)
        # embedding & encoding
        embedding = self.embedding(inputs)
        outputs = self.encoding(embedding)
        # encode
        attentions = []
        masked_attentions = []
        for layer in self.layers:
            outputs, masked_attention, attention = layer(outputs, enc_outputs, masked_atten_mask, enc_atten_mask)
            attentions.append(attention)
            masked_attentions.append(masked_attention)
        return outputs, masked_attentions, attentions

        