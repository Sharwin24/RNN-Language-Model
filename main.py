import torch
import torch.nn as nn
from datasets import load_dataset
from rnn import RNN
import torch.optim as optim

epochs = 20

# Load WikiText-2 dataset using Hugging Face datasets
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

# Access splits
train_dataset = dataset['train']
validation_dataset = dataset['validation']
test_dataset = dataset['test']

print(train_dataset[1]["text"].split(" "))




