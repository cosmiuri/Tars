import chromadb
from sentence_transformers import SentenceTransformer


DB_PATH = "vectorstore"
COLLECTION_NAME = "interstellar_script"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def search_interstellar(question, top_k=5):
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


if __name__ == "__main__":
    question = "how are you tars?"

    results = search_interstellar(question, top_k=5)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    print("QUESTION:")
    print(question)

    for i in range(len(documents)):
        print("=" * 80)
        print(f"RESULT {i + 1}")
        print(f"Page: {metadatas[i]['page']}")
        print(f"Distance: {distances[i]}")
        print("-" * 80)
        print(documents[i][:1000])