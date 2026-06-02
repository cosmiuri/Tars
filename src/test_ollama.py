import requests


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2:3b"


def call_ollama(prompt: str, model: str = MODEL_NAME) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "stream": False,
        },
        timeout=120,
    )

    if response.status_code != 200:
        print("STATUS CODE:", response.status_code)
        print("OLLAMA ERROR:")
        print(response.text)
        response.raise_for_status()

    data = response.json()
    return data["message"]["content"]


if __name__ == "__main__":
    prompt = "Explain geodesics in one short paragraph."
    answer = call_ollama(prompt)

    print("=" * 80)
    print("PROMPT:")
    print(prompt)
    print("=" * 80)
    print("ANSWER:")
    print(answer)