import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from rnn import RNN
import torch.optim as optim
from collections import Counter

VOCAB_SIZE = 15000
epochs = 20
batch_size = 32

# Load WikiText-2 dataset using Hugging Face datasets
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

# Access splits
train_dataset = dataset['train']
validation_dataset = dataset['validation']
test_dataset = dataset['test']


#Create a reduced vocabulary by taking the top 15,000 most common words
word_counter = Counter()
for example in train_dataset:
    # Tokenization by whitespace
    words = example["text"].split(" ")
    word_counter.update(words)
    
most_common_words = word_counter.most_common(VOCAB_SIZE)

#Vocab now contains the most common words
vocab = set()
for word, _ in most_common_words:
    vocab.add(word)

vocab.add('<UNK>')  # For unknown words

#Integer representation
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
            example_indices.append(word_to_idx['<UNK>'])  # Use <UNK> for unknown words
    train_indices.append(example_indices)
    
# Convert indices to tensors
train_tensors = [torch.tensor(seq, dtype=torch.long) for seq in train_indices]

# Combine all sequences into a single tensor
train_data = torch.cat(train_tensors)

# Create a TensorDataset and DataLoader
train_dataset = TensorDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = RNN(len(vocab), embedding_dim=100)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Hyperparameters
sequence_length = 20  # Number of unrolled time steps (e.g., 20)

# Training loop
for epoch in range(epochs):
    model.train()
    hidden = None
    total_loss = 0
    
    for batch in train_loader:
        batch = batch[0]  # (batch_size,)
        
        # Prepare input and target sequences
        inputs = batch[:-1]  # All except the last token
        targets = batch[1:]   # All except the first token
        
        # Forward pass
        outputs, hidden = model(inputs.unsqueeze(0), hidden)
        hidden = hidden.detach()  # Detach hidden state to prevent backprop through time
        
        # Compute loss
        loss = criterion(outputs.view(-1, len(vocab)), targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')







