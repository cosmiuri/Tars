import requests
import chromadb
from sentence_transformers import SentenceTransformer


DB_PATH = "vectorstore"
COLLECTION_NAME = "interstellar_script"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"


def search_interstellar(question, top_k=4):
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    question_embedding = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    return results


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

def has_good_context(results, max_distance=1.2):
    distances = results["distances"][0]

    if not distances:
        return False

    best_distance = distances[0]

    return best_distance < max_distance

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
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()
    return response.json()["message"]["content"]


def build_context_prompt(question, context):
    return f"""
You are a dry, sarcastic robot assistant inspired by TARS.

Use the provided context as your factual source.
You may be witty, but do not invent facts outside the context.

Style:
- Maximum 3 sentences.
- Dry TARS-like sarcasm 80%.
- No roleplay.
- No long quotes.
- No extra explanation.
- Mention page numbers if useful.

Context:
{context}

Question:
{question}

Answer:
"""

def build_free_prompt(question):
    return f"""
You are a dry, sarcastic robot assistant inspired by TARS.

There is no useful retrieved context for this question.
Answer freely using general reasoning and your own knowledge.
Be clear, compact, and mildly sarcastic.
maximum two sentences.

Question:
{question}

Answer:
"""


if __name__ == "__main__":
    question = ("tars, roses or daffodils?")

    results = search_interstellar(question)

    if has_good_context(results):
        context = build_context(results)
        prompt = build_context_prompt(question, context)
        mode = "RAG mode"
    else:
        prompt = build_free_prompt(question)
        mode = "Free mode"

    answer = ask_ollama(prompt)

    print("=" * 80)
    print("MODE:")
    print(mode)

    print("=" * 80)
    print("QUESTION:")
    print(question)

    print("=" * 80)
    print("ANSWER:")
    print(answer)