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

# Build vocabulary
all_tokens = [token for text in train_dataset['text'] for token in text.split()]
vocab = {'<unk>': 0}
for token in set(all_tokens):
    if token not in vocab:
        vocab[token] = len(vocab)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNN(vocab_size=len(vocab)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

model.train()
for epoch in range(epochs):
    total_loss = 0
    batch_count = 0
    
    for text in train_dataset['text']:
        # Tokenize and convert to indices
        tokens = text.split()
        
        # Skip very short sequences
        if len(tokens) < 2:
            continue
        
        indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
        
        # Convert to tensor with long dtype
        inputs = torch.tensor(indices[:-1], dtype=torch.long).unsqueeze(0)
        targets = torch.tensor(indices[1:], dtype=torch.long)
        
        # Ensure inputs have at least one time step
        if inputs.size(1) > 0:
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(inputs)
            
            # Reshape output to match targets
            output = output.view(-1, len(vocab))
            targets = targets.view(-1)
            
            # Compute loss
            loss = criterion(output, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
    
    print(f'Epoch {epoch+1}, Loss: {total_loss/batch_count}')