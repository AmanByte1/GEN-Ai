import requests

url = "http://localhost:11434/api/generate"

while True:
    user_input = input("You: ")

    if user_input == "exit":
        break

    data = {
        "model": "llama3",
        "prompt": user_input,
        "stream": False
    }

    response = requests.post(url, json=data)
    result = response.json()

    print("AI:", result["response"])