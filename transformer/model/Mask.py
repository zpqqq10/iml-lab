import torch

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
    mask = torch.triu(torch.ones(batch_size, seq_len, seq_len), diagonal=1).bool()
    # repeat the mask for batch_size times
    # mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask