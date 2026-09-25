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

def generate_text(
    prompt,
    max_words=50,
    temperature=0.8
):

    words = prompt.lower().split()

    generated = words.copy()

    for _ in range(max_words):

        input_ids = []

        for word in generated[-20:]:

            if word in word_to_id:
                input_ids.append(
                    word_to_id[word]
                )

        if len(input_ids) == 0:
            break

        x = torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=device
        )

        with torch.no_grad():

            logits, _ = model(x)

            last_logits = logits[:, -1, :]

            last_logits = (
                last_logits / temperature
            )

            probs = torch.softmax(
                last_logits,
                dim=-1
            )

            next_id = torch.multinomial(
                probs,
                num_samples=1
            ).item()

        next_word = id_to_word[next_id]

        generated.append(next_word)

    return " ".join(generated)

prompt = input("Enter starting text: ")

result = generate_text(
    prompt,
    num_words=30
)

print("\nGenerated Text:\n")
print(result)