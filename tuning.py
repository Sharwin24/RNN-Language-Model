import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from collections import Counter
from datasets import load_dataset
from rnn import RNN, RNN_HP
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class HyperParams:
    vocab_size: int
    batch_size: int
    seq_length: int
    learning_rate: float
    num_epochs: int
    hidden_dim: int
    num_layers: int
    embedding_dim: int
    dropout: float

    def __repr__(self):
        return f'HP(vocab_size={self.vocab_size}, batch_size={self.batch_size}, seq_len={self.seq_length}, lr={self.learning_rate}, epochs={self.num_epochs}, hl_dim={self.hidden_dim}, num_layers={self.num_layers}, emb_dim={self.embedding_dim}, do={self.dropout})'


class RNN_LLM:
    def __init__(self, train_valid_test_files: tuple[str, str, str], hp: HyperParams):
        self.train_file, self.valid_file, self.test_file = train_valid_test_files
        self.HP = hp
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using Torch device: {self.device}')
        self.training_setup()
        self.model_setup()

    def train(self, debug=True):
        train_losses = []
        for e in range(self.HP.num_epochs):
            print(f"Epoch {e+1}/{self.HP.num_epochs}") if debug else None
            self.model.train()
            # Initialize hidden layers on every epoch
            hidden = torch.zeros(self.HP.num_layers, self.HP.batch_size,
                                 self.HP.hidden_dim).to(self.device)
            epoch_loss = 0.0
            for batch_idx, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                actual_batch_size = x.size(0)
                # Adjust the hidden state to match the actual batch size
                if hidden.size(1) != actual_batch_size:
                    hidden = torch.zeros(self.HP.num_layers, actual_batch_size,
                                         self.HP.hidden_dim).to(self.device)
                print(
                    f"Processing batch {batch_idx + 1}/{len(self.train_loader)}"
                ) if debug else None
                self.optimizer.zero_grad()
                output, hidden = self.model(x, hidden)
                loss = self.loss_func(
                    output.view(-1, self.HP.vocab_size), y.view(-1))
                loss.backward()
                self.optimizer.step()
                # Prevent backprop through time?
                hidden = hidden.detach()
                epoch_loss += loss.item()
                print(
                    f"Batch {batch_idx + 1}/{len(self.train_loader)}, Loss: {loss.item()}"
                ) if debug else None
                # End of batch loop
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            train_losses.append(epoch_loss)
            print(
                f"Epoch {e+1}/{self.HP.num_epochs} Loss: {avg_epoch_loss}"
            ) if debug else None
            # End of epoch loop
        plt.plot(range(1, self.HP.num_epochs + 1), train_losses,
                 label='Training Loss', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig(f'training_loss_{self.HP.num_epochs}_epochs.png')

    def model_setup(self):
        self.model = RNN_HP(self.train_vocab, self.HP)
        self.model = self.model.to(self.device)
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.HP.learning_rate)

    def training_setup(self):
        # Load the data and create a reduced vocabulary
        self.train_indices, self.train_vocab = self.load_data(self.train_file)
        self.valid_indices, self.valid_vocab = self.load_data(self.valid_file)
        self.test_indices, self.test_vocab = self.load_data(self.test_file)

        # Create input-output pairs for the dataset
        self.train_inputs, self.train_targets = self.create_sequences(
            self.train_indices, self.HP.seq_length)
        self.valid_inputs, self.valid_targets = self.create_sequences(
            self.valid_indices, self.HP.seq_length)
        self.test_inputs, self.test_targets = self.create_sequences(
            self.test_indices, self.HP.seq_length)

        # Create the TensorDataset objects
        self.train_dataset = TensorDataset(
            self.train_inputs, self.train_targets)
        self.valid_dataset = TensorDataset(
            self.valid_inputs, self.valid_targets)
        self.test_dataset = TensorDataset(self.test_inputs, self.test_targets)

        # Create the DataLoader objects
        self.train_loader = DataLoader(
            self.train_dataset, self.HP.batch_size, shuffle=True)
        self.valid_loader = DataLoader(
            self.valid_dataset, self.HP.batch_size, shuffle=True)
        self.test_loader = DataLoader(
            self.test_dataset, self.HP.batch_size, shuffle=True)

    def create_sequences(self, data, seq_length):
        inputs = []
        targets = []
        for i in range(len(data) - seq_length):
            inputs.append(data[i:i + seq_length])  # input sequence
            # target sequence, shifted by one
            targets.append(data[i + 1:i + seq_length + 1])
        return torch.tensor(inputs), torch.tensor(targets)

    def load_wikitext(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text

    def tokenize(self, text):
        return text.split(' ')

    def reduce_vocab(self, tokens, threshold=5):
        token_counts = Counter(tokens)
        reduced_tokens = [token if token_counts[token] >=
                          threshold else '<unk>' for token in tokens]
        return reduced_tokens

    def create_vocab_mapping(self, tokens):
        vocab = set(tokens)
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        idx_to_word = {idx: word for word, idx in word_to_idx.items()}
        return word_to_idx, idx_to_word

    def tokens_to_indices(self, tokens, word_to_idx):
        # return a list of indices (based on the dict mapping words to tokens)
        return [word_to_idx[token] for token in tokens]

    def load_data(self, text):
        data_text = self.load_wikitext(text)
        data_tokens = self.tokenize(data_text)
        reduced_data_tokens = self.reduce_vocab(data_tokens)
        data_word_to_idx, data_idx_to_word = self.create_vocab_mapping(
            reduced_data_tokens)
        vocab_size = len(data_word_to_idx)
        data_indices = self.tokens_to_indices(
            reduced_data_tokens, data_word_to_idx)
        return data_indices, vocab_size
