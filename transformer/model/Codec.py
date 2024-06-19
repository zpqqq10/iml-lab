import torch
import torch.nn as nn
from .Attention import MultiAttention
from .FeedForward import FeedForward
from .Embedding import Embeddings
from .Encoding import RoPE
from .Mask import mask_stop_words, mask_subsequence

class EncoderLayer(nn.Module):
    '''one layer of encoder
    '''
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.encoder_atten = MultiAttention()
        self.ffnet = FeedForward()
        
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
        n_layers (int): number of layers
    '''
    def __init__(self, vocab = 1e4, emb_dim = 512, n_layers = 6):
        super(Encoder, self).__init__()
        self.embedding = Embeddings(emb_dim, vocab)
        self.encoding = RoPE(emb_dim)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        
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
    
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.decoder_atten = MultiAttention()
        self.decoder_masked_atten = MultiAttention()
        self.ffnet = FeedForward()
    
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
    
class Decoder(nn.Module):
    '''decoder itself

    Args:
        vocab (int): number of words in the vocabulary
        emb_dim (int): length of the embedding vector
        n_layers (int): number of layers
    '''
    def __init__(self, vocab = 1e4, emb_dim = 512, n_layers = 6):
        super(Decoder, self).__init__()
        self.embedding = Embeddings(emb_dim, vocab)
        self.encoding = RoPE(emb_dim)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
       
    # inputs is the input of the decoder
    # enc_inputs is the input of the encoder 
    # enc_outputs is the output of the encoder 
    def forward(self, inputs, enc_inputs, enc_outputs): 
        # mask out stop words and subsequence in the inputs
        masked_atten_mask = mask_subsequence(inputs)
        masked_atten_mask = masked_atten_mask.bool() & mask_stop_words(inputs, inputs)
        enc_atten_mask = mask_stop_words(enc_inputs, inputs)
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

        