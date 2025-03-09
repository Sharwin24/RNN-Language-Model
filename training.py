import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
from collections import Counter
from rnn import RNN

# Load the wikitext-2 dataset


def load_wikitext(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def tokenize(text):
    return text.split(' ')

# reduce vocabulary size


def reduce_vocab(tokens, threshold=5):
    token_counts = Counter(tokens)
    reduced_tokens = [token if token_counts[token] >=
                      threshold else '<unk>' for token in tokens]
    return reduced_tokens

# create a vocab mapping, from tokens to indices


def create_vocab_mapping(tokens):
    vocab = set(tokens)
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return word_to_idx, idx_to_word

# return a list of indices (based on the dict mapping words to tokens)


def tokens_to_indices(tokens, word_to_idx):
    return [word_to_idx[token] for token in tokens]


def load_data(text):
    data_text = load_wikitext(text)
    data_tokens = tokenize(data_text)
    reduced_data_tokens = reduce_vocab(data_tokens)
    data_word_to_idx, data_idx_to_word = create_vocab_mapping(
        reduced_data_tokens)
    vocab_size = len(data_word_to_idx)
    data_indices = tokens_to_indices(reduced_data_tokens, data_word_to_idx)
    return data_indices, vocab_size


# # load training data
# train_text = load_wikitext('wiki2.train.txt')
# train_tokens = tokenize(train_text)
# reduced_train_tokens = reduce_vocab(train_tokens)
# train_word_to_idx, train_idx_to_word = create_vocab_mapping(reduced_train_tokens)
# train_indices = tokens_to_indices(reduced_train_tokens, train_word_to_idx)
train_indices, train_vocab = load_data('wiki2.train.txt')
val_indices, val_vocab = load_data('wiki2.valid.txt')
test_indices, test_vocab = load_data('wiki2.test.txt')


# creating input output pairs for the dataset
def create_sequences(data, seq_length):
    inputs = []
    targets = []
    for i in range(len(data) - seq_length):
        inputs.append(data[i:i + seq_length])  # input sequence
        # target sequence, shifted by one
        targets.append(data[i + 1:i + seq_length + 1])
    return torch.tensor(inputs), torch.tensor(targets)


# training setup
batch_size = 32
seq_length = 10  # unrolled time steps
learning_rate = 0.001
num_epochs = 5
hidden_dim = 256
num_layers = 2

train_inputs, train_targets = create_sequences(train_indices, seq_length)
val_inputs, val_targets = create_sequences(val_indices, seq_length)
test_inputs, test_targets = create_sequences(test_indices, seq_length)

train_dataset = TensorDataset(train_inputs, train_targets)
val_dataset = TensorDataset(val_inputs, val_targets)
test_dataset = TensorDataset(test_inputs, test_targets)

train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=True)


# vocab_size = len(train_word_to_idx)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = RNN(train_vocab)
model = model.to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print(f"Code is here!")


def init_hidden(batch_size, hidden_dim, num_layers):
    hidden = torch.zeros(num_layers, batch_size, hidden_dim)
    return hidden


train_losses = []

# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    model.train()
    # hidden = None

    hidden = init_hidden(batch_size, hidden_dim, num_layers).to(device)
    epoch_loss = 0.0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        actual_batch_size = x.size(0)

        # Adjust the hidden state to match the actual batch size
        if hidden.size(1) != actual_batch_size:
            hidden = init_hidden(
                actual_batch_size, hidden_dim, num_layers).to(device)
        # print(f"Processing batch {batch_idx + 1}/{len(train_loader)}")
        optimizer.zero_grad()
        output, hidden = model(x, hidden)
        loss = loss_func(output.view(-1, train_vocab), y.view(-1))
        loss.backward()
        optimizer.step()

        # preventing backprop through time?
        hidden = hidden.detach()

        epoch_loss += loss.item()

        # print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item()}")
    avg_epoch_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_epoch_loss)  # Store the average loss

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss}')

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses,
         label='Training Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.show()
