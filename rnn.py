import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=256, num_layers=2):
        super().__init__()  # Remove 'self' from super() call
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # RNN layers
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        # Embed the input
        embedded = self.embedding(x)
        # Pass through RNN
        output, hidden = self.rnn(embedded, hidden)
        # Take the last time step and pass through final layer
        output = self.fc(output)
        return output, hidden