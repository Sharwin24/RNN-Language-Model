import torch
import torch.nn as nn
from datasets import load_dataset
from rnn import RNN
import torch.optim as optim
from collections import Counter

VOCAB_SIZE = 15000
epochs = 20

# Load WikiText-2 dataset using Hugging Face datasets
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

# Access splits
train_dataset = dataset['train']
validation_dataset = dataset['validation']
test_dataset = dataset['test']


# Create a reduced vocabulary by taking the top 15,000 most common words
word_counter = Counter()
for example in train_dataset:
    # Tokenization by whitespace
    words = example["text"].split(" ")
    word_counter.update(words)

most_common_words = word_counter.most_common(VOCAB_SIZE)

# Vocab now contains the most common words
vocab = set()
for word, _ in most_common_words:
    vocab.add(word)

vocab.add('<UNK>')  # For unknown words

# Integer representation
word_to_idx = {}
idx = 0
for word in vocab:
    word_to_idx[word] = idx
    idx += 1

# Convert the training dataset to integer indices
train_indices = []
for example in train_dataset:
    words = example["text"].split(" ")
    example_indices = []
    for word in words:
        if word in word_to_idx:
            example_indices.append(word_to_idx[word])
        else:
            # Use <UNK> for unknown words
            example_indices.append(word_to_idx['<UNK>'])
    train_indices.append(example_indices)

print(train_indices[:10])
