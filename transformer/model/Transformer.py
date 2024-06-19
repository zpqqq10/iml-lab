import torch
import torch.nn as nn
from .Codec import Encoder, Decoder

# https://blog.csdn.net/weixin_42475060/article/details/121101749
# https://mp.weixin.qq.com/s/XFniIyQcrxambld5KmXr6Q
# https://blog.csdn.net/qq_37236745/article/details/107352273
# Annotated transformer

class Transformer(nn.Module):
    def __init__(self, emb_dim = 512, src_vocab = 1e4, tgt_vocab = 1e4):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab)
        self.decoder = Decoder(tgt_vocab)
        self.projection = nn.Linear(emb_dim, tgt_vocab, bias=False)
        self.softmax = nn.Softmax()
        
    def forward(self, inputs, targets):
        enc_outputs, _ = self.encoder(inputs)
        dec_outputs, _, _ = self.decoder(targets, inputs, enc_outputs)
        # softmax is included in CrossEntropyLoss
        outputs = self.projection(dec_outputs)
        return outputs.view(-1, outputs.size(-1))
        