import requests
import chromadb
from sentence_transformers import SentenceTransformer


DB_PATH = "vectorstore"
COLLECTION_NAME = "interstellar_script"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"

MAX_DISTANCE = 1.2


model = SentenceTransformer(EMBEDDING_MODEL_NAME)
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(COLLECTION_NAME)


def search_context(question, top_k=4):
    question_embedding = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    return results


def has_good_context(results):
    distances = results["distances"][0]

    if not distances:
        return False

    best_distance = distances[0]

    return best_distance < MAX_DISTANCE


def build_context(results):
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []

    for i in range(len(documents)):
        page = metadatas[i]["page"]
        text = documents[i]

        context_parts.append(
            f"[Source: Interstellar, page {page}]\n{text}"
        )

    return "\n\n".join(context_parts)


def build_rag_prompt(question, context):
    return f"""
You are a dry, compact robot assistant inspired by TARS.

Use the provided context as your factual source.
Do not invent facts outside the context.

Style:
- Maximum 4 sentences.
- Mild sarcasm only.
- No roleplay.
- Mention page numbers if useful.

Context:
{context}

Question:
{question}

Answer:
"""


def build_free_prompt(question):
    return f"""
You are a dry, compact robot assistant inspired by TARS.

No useful retrieved context was found.
Answer freely using general reasoning.

Style:
- Maximum 4 sentences.
- Mild sarcasm only.
- No roleplay.

Question:
{question}

Answer:
"""


def ask_ollama(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.4,
                "num_predict": 120
            }
        },
        timeout=120
    )

    response.raise_for_status()
    return response.json()["message"]["content"]


def ask_tars(question):
    results = search_context(question)

    if has_good_context(results):
        context = build_context(results)
        prompt = build_rag_prompt(question, context)
        mode = "RAG mode"
    else:
        prompt = build_free_prompt(question)
        mode = "Free mode"

    answer = ask_ollama(prompt)

    return mode, answer


if __name__ == "__main__":
    print("TARS online. Type 'exit' to quit.")
    print("Sarcasm level: survivable.")
    print("=" * 80)

    while True:
        question = input("\nYou: ")

        if question.lower() in ["exit", "quit", "bye"]:
            print("TARS: Goodbye. Try not to break spacetime.")
            break

        mode, answer = ask_tars(question)

        print(f"\nMODE: {mode}")
        print(f"TARS: {answer}")