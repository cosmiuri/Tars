from pathlib import Path

import fitz
import chromadb
from sentence_transformers import SentenceTransformer


PDF_PATH = Path("/Users/cosmiuri/PycharmProjects/physics_carroll_rag/data/interstellar.pdf")
DB_PATH = "vectorstore"
COLLECTION_NAME = "interstellar_script"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def extract_pdf_pages(pdf_path):
    doc = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text().strip()

        if text:
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


def make_chunks(pages, chunk_size=900, overlap=150):
    chunks = []
    metadatas = []
    ids = []

    chunk_id = 0

    for page in pages:
        text = page["text"]
        page_number = page["page"]

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

                metadatas.append({
                    "page": page_number,
                    "source": "Interstellar"
                })

                ids.append(f"chunk_{chunk_id}")
                chunk_id += 1

            start += chunk_size - overlap

    return chunks, metadatas, ids


def save_to_chroma(chunks, metadatas, ids):
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Embedding chunks...")

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=True
    ).tolist()

    client = chromadb.PersistentClient(path=DB_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Saved {len(chunks)} chunks to ChromaDB.")


if __name__ == "__main__":
    pages = extract_pdf_pages(PDF_PATH)
    print(f"Read {len(pages)} pages.")

    chunks, metadatas, ids = make_chunks(pages)
    print(f"Created {len(chunks)} chunks.")

    save_to_chroma(chunks, metadatas, ids)