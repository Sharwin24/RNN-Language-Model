import torch.nn as nn
from tuning import HyperParams


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()  # Remove 'self' from super() call
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # RNN layers
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        # Embed the input
        embedded = self.embedding(x)
        # Pass through RNN
        rnn_out, hidden = self.rnn(embedded, hidden)
        # Take the last time step and pass through final layer
        output = self.fc(rnn_out)
        return output, hidden


class RNN_HP(nn.Module):
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
        # Output layer
        self.fc = nn.Linear(hp.hidden_dim, hp.vocab_size)

    def forward(self, x, hidden):
        # Embed the input
        embedded = self.embedding(x)
        # Pass through RNN
        rnn_out, hidden = self.rnn(embedded, hidden)
        # Take the last time step and pass through final layer
        output = self.fc(rnn_out)
        return output, hidden
