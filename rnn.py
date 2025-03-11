import torch.nn as nn
from dataclasses import dataclass


@dataclass
class HyperParams:
    '''Dataclass to store hyperparameters for the RNN Language Model'''

    vocab_size: int
    batch_size: int
    seq_length: int
    learning_rate: float
    num_epochs: int
    hidden_dim: int
    num_layers: int
    embedding_dim: int
    dropout: float

    def __hash__(self):
        return hash(
            (
                self.vocab_size,
                self.batch_size,
                self.seq_length,
                self.learning_rate,
                self.num_epochs,
                self.hidden_dim,
                self.num_layers,
                self.embedding_dim,
                self.dropout
            )
        )

    def __eq__(self, value: 'HyperParams'):
        return (
            self.vocab_size == value.vocab_size
            and self.batch_size == value.batch_size
            and self.seq_length == value.seq_length
            and self.learning_rate == value.learning_rate
            and self.num_epochs == value.num_epochs
            and self.hidden_dim == value.hidden_dim
            and self.num_layers == value.num_layers
            and self.embedding_dim == value.embedding_dim
            and self.dropout == value.dropout
        )

    def __repr__(self):
        return f'HP(vocab_size={self.vocab_size}, batch_size={self.batch_size}, seq_len={self.seq_length}, lr={self.learning_rate}, epochs={self.num_epochs}, hl_dim={self.hidden_dim}, num_layers={self.num_layers}, emb_dim={self.embedding_dim}, do={self.dropout})'


class RNN(nn.Module):
    def __init__(self, hp: HyperParams):
        super().__init__()  # Remove 'self' from super() call
        self.HP = hp
        # Embedding layer
        self.embedding = nn.Embedding(hp.vocab_size, hp.embedding_dim)
        # RNN layers
        self.rnn = nn.RNN(
            input_size=hp.embedding_dim,
            hidden_size=hp.hidden_dim,
            num_layers=hp.num_layers,
            batch_first=True,
            dropout=hp.dropout
        )
        # Dropout layer
        self.dropout = nn.Dropout(hp.dropout)
        # Output layer
        self.fc = nn.Linear(hp.hidden_dim, hp.vocab_size)

    def forward(self, x, hidden):
        # Embed the input
        embedded = self.embedding(x)
        # Pass through RNN
        rnn_out, hidden = self.rnn(embedded, hidden)
        # Take the last time step and pass through final layer
        logits = self.dropout(rnn_out)
        output = self.fc(logits)
        return output, hidden
