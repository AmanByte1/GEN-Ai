import requests
import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FILE_NAME = "chat_history.json"

# 🔹 Load old chat OR create new with system role
if os.path.exists(FILE_NAME):
    with open(FILE_NAME, "r") as f:
        messages = json.load(f)
else:
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant"}
    ]

print("Chatbot started! Type 'exit' to stop.\n")


# 🔍 Improved RAG function
def get_relevant_messages(messages, query, top_n=2):
    # remove system message from search
    filtered_messages = [m for m in messages if m["role"] != "system"]

    if len(filtered_messages) == 0:
        return []

    texts = [msg["content"] for msg in filtered_messages]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts + [query])

    similarity = cosine_similarity(vectors[-1], vectors[:-1])
    scores = similarity.flatten()

    # get top indices
    top_indices = scores.argsort()[-top_n:]

    # 🔥 FIX: maintain order
    top_indices = sorted(top_indices)

    return [filtered_messages[i] for i in top_indices]


# 🔁 Chat loop
while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    # save user message
    messages.append({"role": "user", "content": user_input})

    # 🔍 get relevant messages
    relevant_messages = get_relevant_messages(messages, user_input)

    # 🧠 build clean prompt
    full_prompt = ""

    # add system role first
    for msg in messages:
        if msg["role"] == "system":
            full_prompt += f"{msg['role']}: {msg['content']}\n"

    # add relevant messages
    for msg in relevant_messages:
        full_prompt += f"{msg['role']}: {msg['content']}\n"

    # add current user input
    full_prompt += f"user: {user_input}\nassistant:"

    # send to ollama
    data = {
        "model": "llama3",
        "prompt": full_prompt,
        "stream": False
    }

    response = requests.post("http://localhost:11434/api/generate", json=data)
    result = response.json()

    reply = result["response"]
    print("AI:", reply)

    # save AI reply
    messages.append({"role": "assistant", "content": reply})

    # save to file
    with open(FILE_NAME, "w") as f:
        json.dump(messages, f, indent=2)