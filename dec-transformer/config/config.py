conf = {
    'exp': 'trial4',
    'lr': 3e-5,
    'lr_decay': 20,
    # length of embedding vector
    'emb_dim': 64,
    'ff_dim': 64*4,
    'heads': 8,
    'encoder_layers': 1,
    'decoder_layers': 2,
    # how many tokens processed at a time
    'context_length': 256,
    'batch_size': 32,
    'iterations': 2e5,
    'ckpt_path': 'ckpt',
    'ckpt_iter': 5e3,
    'log_iter': 1e3,
    # how many iterations to solve validate loss
    'val_iterations': 1e2,
    
}