conf = {
    'exp': 'base',
    'lr': 1e-4,
    'lr_decay': 500,
    'decay_initiation': 0.4,
    # length of embedding vector
    'emb_dim': 512,
    'ff_dim': 512*4,
    'heads': 8,
    'encoder_layers': 8,
    'decoder_layers': 8,
    # how many tokens processed at a time
    'context_length': 256,
    'batch_size': 32,
    'iterations': 8e4,
    'ckpt_path': 'ckpt',
    'ckpt_iter': 5e3,
    'log_iter': 1e3,
    # how many iterations to solve validate loss
    'val_iterations': 1e2,
    
}