import torch
import torch.nn as nn
import pandas as pd

# Setup torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Setup dataset
raw_data = pd.read_csv('./animes.csv')
raw_data = list(raw_data['synopsis'])
