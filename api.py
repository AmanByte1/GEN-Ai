from fastapi import FastAPI
import requests
import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

FILE_NAME = "chat_history.json"

# Load messages
if os.path.exists(FILE_NAME):
    with open(FILE_NAME, "r") as f:
        messages = json.load(f)
else:
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant"}
    ]


# RAG function
def get_relevant_messages(messages, query, top_n=2):
    filtered_messages = [m for m in messages if m["role"] != "system"]

    if len(filtered_messages) == 0:
        return []

    texts = [msg["content"] for msg in filtered_messages]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts + [query])

    similarity = cosine_similarity(vectors[-1], vectors[:-1])
    scores = similarity.flatten()

    top_indices = scores.argsort()[-top_n:]
    top_indices = sorted(top_indices)

    return [filtered_messages[i] for i in top_indices]


# 🔥 API endpoint
@app.post("/ask")
def ask_ai(data: dict):
    user_input = data["question"]

    messages.append({"role": "user", "content": user_input})

    relevant_messages = get_relevant_messages(messages, user_input)

    full_prompt = ""

    # system
    for msg in messages:
        if msg["role"] == "system":
            full_prompt += f"{msg['role']}: {msg['content']}\n"

    # relevant
    for msg in relevant_messages:
        full_prompt += f"{msg['role']}: {msg['content']}\n"

    full_prompt += f"user: {user_input}\nassistant:"

    # call ollama
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": full_prompt,
            "stream": False
        }
    )

    result = response.json()
    reply = result["response"]

    messages.append({"role": "assistant", "content": reply})

    # save
    with open(FILE_NAME, "w") as f:
        json.dump(messages, f, indent=2)

    return {"answer": reply}