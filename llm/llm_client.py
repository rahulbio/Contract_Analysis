import requests

def ollama_client(prompt, model="llama3.2:3b"):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )
    r.raise_for_status()
    data = r.json()

    # Robust extraction (works across Ollama versions)
    if "response" in data:
        return data["response"]
    if "message" in data and "content" in data["message"]:
        return data["message"]["content"]

    return str(data)
