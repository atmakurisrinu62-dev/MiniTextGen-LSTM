import torch
import torch.nn as nn
import pickle

from torch.utils.data import DataLoader, TensorDataset
from model import LSTMTextModel


# -------------------------
# Device
# -------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# -------------------------
# Hyperparameters
# -------------------------
sequence_length = 20
embedding_dim = 64
hidden_size = 128
batch_size = 32
epochs = 20
learning_rate = 0.001


# -------------------------
# Load vocabulary
# -------------------------
with open("word_to_id.pkl", "rb") as f:
    word_to_id = pickle.load(f)

with open("id_to_word.pkl", "rb") as f:
    id_to_word = pickle.load(f)

vocab_size = len(word_to_id)

print("Vocabulary size:", vocab_size)


# -------------------------
# Load Tiny Shakespeare
# -------------------------
with open(
    "tiny_shakespeare.txt",
    "r",
    encoding="utf-8"
) as f:
    text = f.read()

words = text.lower().split()

print("Total words:", len(words))


# -------------------------
# Create sequences
# -------------------------
X = []
Y = []

for i in range(
    len(words) - sequence_length
):

    input_words = words[
        i:i + sequence_length
    ]

    target_word = words[
        i + sequence_length
    ]

    try:
        input_ids = [
            word_to_id[word]
            for word in input_words
        ]

        target_id = word_to_id[
            target_word
        ]

        X.append(input_ids)
        Y.append(target_id)

    except KeyError:
        continue


X = torch.tensor(
    X,
    dtype=torch.long
)

Y = torch.tensor(
    Y,
    dtype=torch.long
)

print("X shape:", X.shape)
print("Y shape:", Y.shape)


# -------------------------
# DataLoader
# -------------------------
dataset = TensorDataset(X, Y)

train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)


# -------------------------
# Model
# -------------------------
model = LSTMTextModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_size=hidden_size
).to(device)


# -------------------------
# Loss + Optimizer
# -------------------------
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)


# -------------------------
# Training
# -------------------------
for epoch in range(epochs):

    model.train()

    total_loss = 0

    for x, y in train_loader:

        x = x.to(device)
        y = y.to(device)

        output, _ = model(x)

        # only last sequence position
        logits = output[:, -1, :]

        loss = criterion(
            logits,
            y
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = (
        total_loss /
        len(train_loader)
    )

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Loss: {avg_loss:.4f}"
    )


# -------------------------
# Save trained model
# -------------------------
torch.save(
    model.state_dict(),
    "lstm_shakespeare_local.pth"
)

print("Model saved.")