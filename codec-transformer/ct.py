import torch
import torch.nn as nn
from transformers import LlamaTokenizer
from icecream import ic
from config.config import conf
from model.Transformer import Transformer

def tokenize(path: str = "input.txt"):
    tokenizer = LlamaTokenizer.from_pretrained("tokenizer.model")

    # open input.txt and tokenize it
    with open(path, "r") as f:
        text = f.read()
        raw_tokens = tokenizer.tokenize(text)

    tokens = set(raw_tokens)
    vocab = {}
    reverse_vocab = {}
    # padding
    vocab['<pad>'] = 0
    reverse_vocab[0] = '<pad>'
    # beginning of sequence
    vocab['<s>'] = 1
    reverse_vocab[1] = '<s>'
    for id, token in enumerate(tokens):
        # leave 0 for padding
        vocab[token] = id + 2
        reverse_vocab[id + 2] = token
    raw_tokens.insert(0, '<s>')
    i = 0
    while i < len(raw_tokens) - 1:
        if raw_tokens[i] == '<0x0A>' and raw_tokens[i + 1] == '<0x0A>':
            # indicate the beginning of a new sequence
            raw_tokens[i + 1] = '<s>'
            i += 1
        i += 1
        
    id_text = torch.LongTensor([vocab[token] for token in raw_tokens])
    print(f'{len(vocab)} tokens in total')
    breakpoint()
            
    return vocab, reverse_vocab, id_text

vocab, reverse_vocab, id_text = tokenize()
model = Transformer(conf['emb_dim'], len(vocab), len(vocab), 'cpu')
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
prs = model.parameters()
print(len([p for p in prs]))
nprs = model.named_parameters()
nprs = {name: param.numel() for name, param in nprs}
print(len(nprs.keys()))
print(nprs)
