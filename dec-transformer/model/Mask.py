import torch

'''
True stands for those should be left: https://github.com/hyunwoongko/transformer/tree/master
1. stop words should be set to False
2. previous tokens should be set to True
3. mask that == 0 should be set to negative 
4. Decoder should use &
'''

'''
True stands for those should be removed: https://github.com/graykode/nlp-tutorial/tree/master
1. stop words should be set to True
2. previous tokens should be set to False
3. mask that == 1 should be set to negative 
4. Decoder should use |
'''

# mask out stop words
# position of stop words will be False
def mask_stop_words(k, q):
    # get the size of the sequence
    batch_size, k_len = k.size()
    q_len = q.size(1)
    # create a mask for stop words(0)
    mask = (k != 0).unsqueeze(1).expand(batch_size, q_len, k_len)
    return mask


# mask out previous tokens by a triangular matrix
# position of previous tokens will be True
def mask_subsequence(sequence):
    # get the size of the sequence
    batch_size, seq_len = sequence.size()
    # create a triangular matrix
    mask = torch.tril(torch.ones(batch_size, seq_len, seq_len)).bool()
    # mask = torch.triu(torch.ones(batch_size, seq_len, seq_len), diagonal=1).bool()
    return mask