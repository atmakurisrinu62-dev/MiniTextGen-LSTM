import torch
import pickle

from torch.utils.data import DataLoader, TensorDataset, random_split
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


# -------------------------
# Load vocabulary
# -------------------------
with open("word_to_id.pkl", "rb") as f:
    word_to_id = pickle.load(f)

vocab_size = len(word_to_id)

print("Vocabulary size:", vocab_size)


# -------------------------
# Load dataset text
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

for i in range(len(words) - sequence_length):

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

    except KeyError:
        continue

    X.append(input_ids)
    Y.append(target_id)


# -------------------------
# Convert to tensors
# -------------------------
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
# Full dataset
# -------------------------
dataset = TensorDataset(
    X,
    Y
)


# -------------------------
# 90 / 10 split
# -------------------------
train_size = int(
    0.9 * len(dataset)
)

val_size = (
    len(dataset) - train_size
)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)


# -------------------------
# DataLoaders
# -------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)


# -------------------------
# Create model
# -------------------------
model = LSTMTextModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_size=hidden_size
).to(device)


# -------------------------
# Load saved weights
# -------------------------
model.load_state_dict(
    torch.load(
        "lstm_shakespeare.pth",
        map_location=device
    )
)

model.eval()

print("Model loaded successfully")


# -------------------------
# Accuracy function
# -------------------------
def calculate_accuracy(
    model,
    data_loader,
    device
):

    correct = 0
    total = 0

    with torch.no_grad():

        for x, y in data_loader:

            x = x.to(device)
            y = y.to(device)

            output, _ = model(x)

            # Last time-step prediction
            logits = output[:, -1, :]

            predictions = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                predictions == y
            ).sum().item()

            total += y.size(0)

    accuracy = (
        correct / total
    ) * 100

    return accuracy


# -------------------------
# Calculate accuracies
# -------------------------
train_accuracy = calculate_accuracy(
    model,
    train_loader,
    device
)

val_accuracy = calculate_accuracy(
    model,
    val_loader,
    device
)


# -------------------------
# Results
# -------------------------
print(
    f"Train Accuracy: {train_accuracy:.2f}%"
)

print(
    f"Validation Accuracy: {val_accuracy:.2f}%"
)