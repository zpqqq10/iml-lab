import torch
import torch.nn as nn
from transformers import LlamaTokenizer
from icecream import ic
from config.config import conf
from model.Transformer import Transformer
import torch.utils.data as Data
import os
from tqdm import tqdm, trange
import shutil
from torch.utils.tensorboard import SummaryWriter
from ordered_set import OrderedSet

# wget -c https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# git clone https://github.com/facebookresearch/llama
# apply as the guidance in the README.md
# bash llama/download.sh

# https://github.com/bl0nder/makespeare/blob/main/makespeare.py
def tokenize(path: str = "input.txt"):
    tokenizer = LlamaTokenizer.from_pretrained("tokenizer.model")
    # padding and beginning of sequence
    # special_tokens = ['<s>']
    special_tokens = ['<s>', '</s>']
    # '<unk>': 0  '<s>': 1  '</s>': 2  '<0x0A>': 13(carriage return)
    
    # convert_tokens_to_string
    # (Pdb) tokenizer('<s>cdc</s>')
    # {'input_ids': [1, 1, 274, 13891, 2], 'attention_mask': [1, 1, 1, 1, 1]}
    # (Pdb) tokenizer('<s>cdc</s>', add_special_tokens=False)
    # {'input_ids': [1, 274, 13891, 2], 'attention_mask': [1, 1, 1, 1]}

    # open input.txt and tokenize it
    with open(path, "r") as f:
        text = f.read()
        raw_tokens = tokenizer.tokenize(text)
    # to ensure the set is in the same order
    tokens = OrderedSet(raw_tokens)
    for tk in special_tokens: 
        tokens.discard(tk)
    vocab = {}
    reverse_vocab = {}
    for i, tk in enumerate(special_tokens):
        vocab[tk] = i
        reverse_vocab[i] = tk
    for id, token in enumerate(tokens):
        # leave 0 for padding
        vocab[token] = id + len(special_tokens)
        reverse_vocab[id + len(special_tokens)] = token
    raw_tokens.insert(0, '<s>')
    # this kind of data does not need this kind of separation
    i = 0
    while i < len(raw_tokens) - 1:
        if raw_tokens[i] == '<0x0A>' and raw_tokens[i + 1] == '<0x0A>':
            # indicate the beginning of a new sequence
            raw_tokens[i] = '</s>'
            raw_tokens[i + 1] = '<s>'
            i += 1
        i += 1
    if raw_tokens[-1] == '<s>':
        # raw_tokens[-1] = '<0x0A>'
        raw_tokens.pop()
        
    id_text = torch.LongTensor([vocab[token] for token in raw_tokens])
    print(f'{len(vocab)} tokens in total')
    print(f'{len(id_text) /1e6} M input tokens in total')
            
    return vocab, reverse_vocab, id_text

# decode a list of ids to a readable string
def decode(ids, reverse_vocab):
    tokenizer = LlamaTokenizer.from_pretrained("tokenizer.model", add_prefix_space=False)
    res = ''.join([reverse_vocab[id.item()] for id in ids[0]])
    res = tokenizer.convert_tokens_to_string(res)
    res = res.replace('<0x0A>', '\n')
    res = res.replace('</s>', '\n')
    res = res.replace('<s>', '\n')
    return res

class TinyDataset(Data.Dataset):
    def __init__(self, inputs, context_length, pad_idx = 0):
        super(TinyDataset, self).__init__()
        self.inputs = inputs
        self.context_length = context_length
        self.pad_idx = pad_idx
  
    def __len__(self):
        return self.inputs.size(0)
    
    def __getitem__(self, idx):
        x = self.inputs[idx: idx+self.context_length]
        y = self.inputs[idx+1: idx+self.context_length+1]
        # padding
        if x.size(0) < self.context_length:
            x = torch.cat([x, torch.full([self.context_length - x.size(0)], self.pad_idx, dtype=torch.long)])
        if y.size(0) < self.context_length:
            y = torch.cat([y, torch.full([self.context_length - y.size(0)], self.pad_idx, dtype=torch.long)])
            # y = torch.cat([y, torch.zeros(self.context_length - y.size(0), dtype=torch.long)])
        return x, y

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab, reverse_vocab, id_text = tokenize()
    split_idx = int(len(id_text) * 0.9)
    train_data = id_text[:split_idx]
    val_data = id_text[split_idx:]
    
    model = Transformer(emb_dim=conf['emb_dim'], ff_dim=conf['ff_dim'], context_len=conf['context_length'],
                        dec_layers=conf['decoder_layers'], n_heads=conf['heads'],
                        vocab=len(vocab), device=device).to(device)
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    print((sum(p.numel() for p in model.parameters()) - conf['emb_dim'] * len(vocab))/1e6, 'M non-embedding parameters')
    criterion = nn.CrossEntropyLoss()
    # 5e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf['lr'])
    # 1e-3
    # optimizer = torch.optim.SGD(model.parameters(), lr=conf['lr'], momentum=0.99)
    
    train_loader = Data.DataLoader(TinyDataset(train_data, conf['context_length'], vocab['<0x0A>']), conf['batch_size'], True)
    valid_loader = Data.DataLoader(TinyDataset(val_data, conf['context_length'], vocab['<0x0A>']), conf['batch_size'], True)
    os.makedirs(os.path.join(conf['ckpt_path'], conf['exp']), exist_ok=True)
    # print('Start training...')
    # shutil.copyfile('config/config.py', os.path.join(conf['ckpt_path'], conf['exp'], 'config.py'))
    # writer = SummaryWriter(os.path.join('logs', conf['exp']))
    # for idx in trange(1, int(conf['iterations']) + 1):
    #     inputs, targets = next(iter(train_loader))
    #     inputs, targets = inputs.to(device), targets.to(device)
    #     optimizer.zero_grad()
    #     outputs = model(inputs)
    #     loss = criterion(outputs, targets.view(-1))
    #     loss.backward()
    #     optimizer.step()
        
    #     # lr decay
    #     decay_steps = conf['lr_decay'] * 1000
    #     decay_factor = 0.1 ** (1 / decay_steps)
    #     if idx > int(conf['iterations'] * conf['decay_initiation']):
    #         for param_group in optimizer.param_groups:
    #             param_group['lr'] = param_group['lr'] * decay_factor
            
    #     if idx % (int(conf['log_iter']) / 10) == 0:
    #         writer.add_scalar('loss/loss', loss, idx)
    #         writer.add_scalar('lr', optimizer.param_groups[0]['lr'], idx)
            
    #     if idx % int(conf['log_iter']) == 0:
    #         val_loss = 0
    #         with torch.no_grad():
    #             model.eval()
    #             for i in range(int(conf['val_iterations'])):
    #                 x, y = next(iter(valid_loader))
    #                 x, y = x.to(device), y.to(device)
    #                 outputs = model(x)
    #                 val_loss += criterion(outputs, y.view(-1))
    #             writer.add_scalar('loss/val_loss', val_loss / conf['val_iterations'], idx)
    #             model.train()
    #         tqdm.write(f'Iteration: {idx} loss = {loss:.8f} val_loss = {val_loss / conf["val_iterations"]:.8f}')
            
    #     if idx % int(conf['ckpt_iter']) == 0:
    #         torch.save(model.state_dict(), os.path.join(conf['ckpt_path'], conf['exp'], f'model_{idx}.pt'))
      
    # writer.close()
    # print('Done!')
    
    # generate
    ckpt = torch.load(os.path.join(conf['ckpt_path'], conf['exp'], f'model_30000.pt'))
    # ckpt = torch.load(os.path.join(conf['ckpt_path'], conf['exp'], f'model_{int(conf["iterations"])}.pt'))
    model.load_state_dict(ckpt)
    # randomly pick one word as the initial input
    model.eval()
    inputs = torch.LongTensor([[vocab['<s>'], vocab['▁C'], vocab['AM'], vocab['ILL'], vocab['O'], vocab[':']]]).to(device)
    result = model.generate(inputs, 64, vocab['</s>'])
    res = decode(result, reverse_vocab)
    print(res)
    with open(os.path.join(conf['ckpt_path'], conf['exp'], f'output_{int(conf["iterations"])}.txt'), 'a+') as f:
        f.write(res)
        f.write('\n----------------------------------------------\n')
    
    