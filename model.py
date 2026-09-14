import torch
import torch.nn as nn


class LSTMTextModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc1 = nn.Linear(
            hidden_size,
            vocab_size
        )

    def forward(self, x, hidden=None):

        embedded = self.embedding(x)

        output, hidden = self.lstm(
            embedded,
            hidden
        )

        logits = self.fc1(output)

        return logits, hidden