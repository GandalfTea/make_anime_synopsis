import torch
import torch.nn as nn
import pandas as pd
import random

# Setup torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# hyperparameters
block_size = 256
batch_size = 64
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embed = 384
n_head = 6
n_layers = 6
dropout = 0.2

torch.manual_seed(1337)

# Setup dataset
raw_data = pd.read_csv('./animes.csv')
raw_data = list(raw_data['synopsis'])
raw_data = [str(i).replace('\r', '').split('\n')[0] for i in raw_data]

raw_data = [f" {i}" for i in raw_data]
raw_data = [ bytes(i, 'utf-8').decode('utf-8', 'ignore') for i in raw_data]
text = ''.join(raw_data)

# Dictionary
vocab = sorted(list(set(''.join(text).split())))
stoi = {s:i+1 for i, s in enumerate(vocab)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
vocab_size=len(vocab)

data = torch.tensor(encode(text), dtype=torch.long)
train_data = data[:int(0.9*len(text))]
val_data   = data[int(0.9*len(text)):]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    t = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), t.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters):
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super()__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)



class Transformer(nn.Module):
    def __init__(self, vocab_size):
        super().__innit__()

