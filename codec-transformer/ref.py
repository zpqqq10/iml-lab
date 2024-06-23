#Import important libraries
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

#Define hyperparameters - These are the hyperparameters that have given me the best result yet
batch_size = 32  #Num of sentences being processed in parallel
context_length = 256  #Num of tokens processed at a time (how much context is there behind understanding each token)
embedding_len = 128 #Each token is converted into an embedding_len dimensional tensor once it undergoes embedding
num_heads = 8 #Num of heads that the embedding matrices will be split in while computing attention
num_encoder_blocks = 1 
num_decoder_blocks = 2  
learning_rate = 5e-5  
max_iterations = 150000 #Num of iterations for which model is trained
eval_interval = 500 #Num of iterations after which validation loss is computed (during model training)
val_iterations = 200 
checkpoint_interval = 10000 #Num of iterations after which a checkpoint is created
num_generated_tokens = 10000  #Num of tokens generated from a trained model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


#Read the dataset
with open('shakespeare_input.txt', 'r', encoding='utf-8') as f:
    input_text = f.read()

#------------TOKENISATION------------#
#Character-Level Tokenization
char_list = sorted(list(set(input_text)))
vocab_size = len(char_list)

char_to_token = {}
token_to_char = {}

for i,c in enumerate(char_list):
  char_to_token[c] = i
  token_to_char[i] = c

#Function to encode string into tokens
def encode(string):
  tokens = []
  for c in string:
    tokens.append(char_to_token[c])
  return tokens

#Function to decode tokens into corresponding characters
def decode(tokens):
  chars = []
  for i in tokens:
    chars.append(token_to_char[i])
  return ''.join(chars)

#Convert token array to tensor for further processing
token_ids = torch.tensor(encode(input_text))

#Train/val split
train_idx = int(len(token_ids)*0.9)
train_data = token_ids[0:train_idx]
val_data = token_ids[train_idx:]

#------------MINI-BATCH SELECTION------------#
def minibatch(train_data, val_data, context_length, batch_size, train=True):

  #Selecting whether to sample from training or validation data
  if (train):
    data = train_data
  else:
    data = val_data

  #Random index to pick minibatch from
  ind = torch.randint(0, len(data) - context_length, size = (batch_size,))

  #Create minibatch
  x_batch = torch.stack([data[i : i+context_length] for i in ind])  #Tokens
  y_batch = torch.stack([data[i+1 : i+context_length+1] for i in ind])  #Next tokens in sentence

  x_batch = x_batch.to(DEVICE)
  y_batch = y_batch.to(DEVICE)

  return x_batch, y_batch

#------------TRAINING------------#
#Function to compute validation loss
@torch.no_grad()
def val_loss (model, val_iterations):
  with torch.no_grad():
    out = {
        'train' : 0,
        'val' : 0
    }
    model.eval()

    for i in range(2):
      for j in range(val_iterations):
        if (i == 0):
          x,y = minibatch(train_data, val_data, context_length, batch_size)
        else:
          x,y = minibatch(train_data, val_data, context_length, batch_size, train=False)

        x = x.to(DEVICE)
        y = y.to(DEVICE)
        logits, cross_entropy_loss = model(x,y)

        if (i==0):
          out['train'] += cross_entropy_loss
        else:
          out['val'] += cross_entropy_loss

    out['train'] /= val_iterations
    out['val'] /= val_iterations

    model.train()
    return out

transformer = Transformer().to(DEVICE)
optimizer = torch.optim.AdamW(transformer.parameters(), lr=learning_rate)
print(sum(p.numel() for p in transformer.parameters())/1e6, 'M parameters') #Number of params in model

#Training loop
for i in range(max_iterations):
  
  #After every eval_interval iterations, compute validation loss
  if (i+1) % eval_interval == 0:
    losses = val_loss(transformer, val_iterations)
    print(f"step {i+1}: train loss {losses['train']}, val loss {losses['val']}")

  #Every checkpoint_interval iterations, create a checkpoint for the model, i.e, save the model state dictionary (along with other info if you want) somewhere
  if ((i+1) % checkpoint_interval == 0):
    checkpoint = {
    'iterations': i+1,
    'num_encoder_blocks': num_encoder_blocks,
    'num_decoder_blocks': num_decoder_blocks,
    'state_dict': transformer.state_dict()  #Most important thing to save
    }
    torch.save(checkpoint, f'models/checkpoint_ctx{context_length}_iter{i+1}_character_encoding.pth')

  #Get minibatch of training data and compute loss
  x, y = minibatch(train_data, val_data, context_length, batch_size, True)
  logits, loss = transformer(x, y)

  #Learn
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

#------------TEXT GENERATION------------#
#Using a pre-trained model by loading a checkpoint
model = Transformer().to(DEVICE)
state_dict = torch.load('models/checkpoint_ctx256_iter150000_character_encoding.pth') #Load saved model  

#When I trained the model, I had an embedding layer in the Decoder class instead of the Transformer class, which I have changed since then. In order for the model to work, 2 of the keys need to be renamed. 
#Comment the following 4 lines if another model is trained.
state_dict['state_dict']['input_embedding.embedding_layer.weight'] = state_dict['state_dict']['decoder.input_embedding.embedding_layer.weight']
state_dict['state_dict']['input_embedding.pos_embedding_layer.weight'] = state_dict['state_dict']['decoder.input_embedding.pos_embedding_layer.weight']
del state_dict['state_dict']['decoder.input_embedding.embedding_layer.weight']
del state_dict['state_dict']['decoder.input_embedding.pos_embedding_layer.weight']

model.load_state_dict(state_dict['state_dict']) #Load state dictionary into model

#Generating Shakespearean text
context = torch.ones((batch_size,context_length), dtype=torch.long, device=DEVICE)
context *= 8  #Token for full-stop
gen_output = decode(model.generate(context, max_new_tokens = num_generated_tokens)[0].tolist())