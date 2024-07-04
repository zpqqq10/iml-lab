import torch
import torch.nn as nn
from .Decoder import Decoder

# https://blog.csdn.net/weixin_42475060/article/details/121101749
# https://mp.weixin.qq.com/s/XFniIyQcrxambld5KmXr6Q
# https://blog.csdn.net/qq_37236745/article/details/107352273
# Annotated transformer

class Transformer(nn.Module):
    def __init__(self, emb_dim = 512, ff_dim=2048, context_len=256,
                 dec_layers = 6, n_heads = 8,
                 vocab = 1e4, device = 'cuda'):
        super(Transformer, self).__init__()
        self.decoder = Decoder(vocab, emb_dim, ff_dim, context_len, dec_layers, n_heads, device)
        self.projection = nn.Linear(emb_dim, vocab, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.context_len = context_len
        
    def forward(self, inputs):
        dec_outputs = self.decoder(inputs)
        # softmax is included in CrossEntropyLoss
        # 2 * d_model * len * vocab
        outputs = self.projection(dec_outputs)
        return outputs.view(-1, outputs.size(-1))
    
    def generate(self, context, max_len = 200, terminate = None):
        for _ in range(max_len):
            dec_outputs = self.decoder(context[:, -self.context_len:])
            outputs = self.projection(dec_outputs)
            outputs = outputs[:, -1, :]
            outputs = self.softmax(outputs)
            # sample one token id as the next
            next_token = torch.multinomial(outputs, 1)
            context = torch.cat([context, next_token], dim=1)
            # terminiate if the token is the end token
            if terminate is not None and next_token.item() == terminate:
                break
        return context
        