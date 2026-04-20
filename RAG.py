import requests
import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FILE_NAME = "chat_history.json"

# Load old chat
if os.path.exists(FILE_NAME):
    with open(FILE_NAME, "r") as f:
        content = f.read().strip()
        messages = json.loads(content) if content else []
else:
    messages = []

print("Chatbot started! Type 'exit' to stop.\n")


# 🔍 RAG function
def get_relevant_messages(messages, query, top_n=3):
    if len(messages) == 0:
        return []

    texts = [msg["content"] for msg in messages]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts + [query])

    similarity = cosine_similarity(vectors[-1], vectors[:-1])
    scores = similarity.flatten()

    top_indices = scores.argsort()[-top_n:]

    return [messages[i] for i in top_indices]


# 🔁 Chat loop
while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    # Save user message
    messages.append({"role": "user", "content": user_input})

    # 🔍 Get relevant past messages (RAG)
    relevant_messages = get_relevant_messages(messages, user_input)

    # Build prompt
    full_prompt = ""
    for msg in relevant_messages:
        full_prompt += f"{msg['role']}: {msg['content']}\n"

    full_prompt += f"user: {user_input}\nassistant:"

    # Send request to Ollama
    data = {
        "model": "llama3",
        "prompt": full_prompt,
        "stream": False
    }

    response = requests.post("http://localhost:11434/api/generate", json=data)
    result = response.json()

    reply = result["response"]
    print("AI:", reply)

    # Save AI reply
    messages.append({"role": "assistant", "content": reply})

    # Save to file
    with open(FILE_NAME, "w") as f:
        json.dump(messages, f, indent=2)