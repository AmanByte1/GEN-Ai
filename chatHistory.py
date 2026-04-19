import requests
import json
import os

fs="chat_history.json"
if os.path.exists(fs):
    with open (fs,"r") as f:
        messages=json.load(f)
else:
    messages=[]
print("Chatbot started! Type 'exit' to stop.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    messages.append({"role": "user", "content": user_input})

    # Combine messages into prompt
    full_prompt = ""
    for msg in messages:
        full_prompt += f"{msg['role']}: {msg['content']}\n"

    data = {
        "model": "llama3",
        "prompt": full_prompt,
        "stream": False
    }

    response = requests.post("http://localhost:11434/api/generate", json=data)
    result = response.json()

    reply = result["response"]
    print("AI:", reply)

    messages.append({"role": "assistant", "content": reply})

    # Save chat
    with open(fs, "w") as f:
        json.dump(messages, f)