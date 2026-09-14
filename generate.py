import torch
import pickle

from model import LSTMTextModel


# Load vocabulary
with open("word_to_id.pkl", "rb") as f:
    word_to_id = pickle.load(f)

id_to_word = {
    idx: word
    for word, idx in word_to_id.items()
}

vocab_size = len(word_to_id)

embedding_dim = 64
hidden_size = 128


model = LSTMTextModel(
    vocab_size,
    embedding_dim,
    hidden_size
)


model.load_state_dict(
    torch.load(
        "lstm_shakespeare.pth",
        map_location="cpu"
    )
)

model.eval()

print("Model loaded successfully")


def generate_text(start_text, num_words=20):

    words = start_text.lower().split()

    input_ids = [
        word_to_id[word]
        for word in words
        if word in word_to_id
    ]

    if len(input_ids) == 0:
        return "Starting words vocabulary lo levu."

    input_tensor = torch.tensor(
        [input_ids],
        dtype=torch.long
    )

    hidden = None

    generated_words = words.copy()

    with torch.no_grad():

        # First, process the complete starting text
        logits, hidden = model(
            input_tensor,
            hidden
        )

        current_id = input_ids[-1]

        for _ in range(num_words):

            current_tensor = torch.tensor(
                [[current_id]],
                dtype=torch.long
            )

            logits, hidden = model(
                current_tensor,
                hidden
            )

            last_logits = logits[:, -1, :]

            predicted_id = torch.argmax(
                last_logits,
                dim=-1
            ).item()

            predicted_word = id_to_word[predicted_id]

            generated_words.append(predicted_word)

            current_id = predicted_id

    return " ".join(generated_words)

prompt = input("Enter starting text: ")

result = generate_text(
    prompt,
    num_words=30
)

print("\nGenerated Text:\n")
print(result)