import torch
import pickle

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import LSTMTextModel


# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# -------------------------------------------------
# Load Vocabulary
# -------------------------------------------------
with open("word_to_id.pkl", "rb") as f:
    word_to_id = pickle.load(f)

with open("id_to_word.pkl", "rb") as f:
    id_to_word = pickle.load(f)


# safer mapping
id_to_word = {
    idx: word
    for word, idx in word_to_id.items()
}

vocab_size = len(word_to_id)

print("Vocabulary size:", vocab_size)


# -------------------------------------------------
# Model Configuration
# -------------------------------------------------
embedding_dim = 64
hidden_size = 128


# -------------------------------------------------
# Create Model
# -------------------------------------------------
model = LSTMTextModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_size=hidden_size
).to(device)


# -------------------------------------------------
# Load Trained Model
# -------------------------------------------------
MODEL_PATH = "lstm_shakespeare.pth"

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Model loaded successfully")


# -------------------------------------------------
# FastAPI App
# -------------------------------------------------
app = FastAPI(
    title="MiniTextGen-LSTM API",
    description="Shakespeare-style text generation using PyTorch LSTM",
    version="1.0.0"
)


# -------------------------------------------------
# CORS
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# -------------------------------------------------
# Request Model
# -------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str
    max_words: int = 50
    temperature: float = 0.8


# -------------------------------------------------
# Text Generation Function
# -------------------------------------------------
def generate_text(
    prompt,
    max_words=50,
    temperature=0.8
):

    # safety
    temperature = max(
        temperature,
        0.1
    )

    max_words = max(
        1,
        min(max_words, 200)
    )

    words = prompt.lower().split()

    if len(words) == 0:
        return ""

    generated = words.copy()

    for _ in range(max_words):

        # use last 20 words
        context_words = generated[-20:]

        input_ids = []

        for word in context_words:

            if word in word_to_id:
                input_ids.append(
                    word_to_id[word]
                )

        # if prompt contains no known words
        if len(input_ids) == 0:
            break

        x = torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=device
        )

        with torch.no_grad():

            logits, _ = model(x)

            # last token prediction
            last_logits = logits[:, -1, :]

            # temperature
            last_logits = (
                last_logits / temperature
            )

            probs = torch.softmax(
                last_logits,
                dim=-1
            )

            # sample next word
            next_id = torch.multinomial(
                probs,
                num_samples=1
            ).item()

        next_word = id_to_word.get(
            next_id,
            ""
        )

        if not next_word:
            break

        generated.append(
            next_word
        )

    return " ".join(generated)


# -------------------------------------------------
# Home Route
# -------------------------------------------------
@app.get("/")
def home():

    return {
        "message": "MiniTextGen-LSTM API is running",
        "device": str(device),
        "vocab_size": vocab_size
    }


# -------------------------------------------------
# Generate Route
# -------------------------------------------------
@app.post("/generate")
def generate(
    request: GenerateRequest
):

    output = generate_text(
        prompt=request.prompt,
        max_words=request.max_words,
        temperature=request.temperature
    )

    return {
        "prompt": request.prompt,
        "max_words": request.max_words,
        "temperature": request.temperature,
        "generated_text": output
    }