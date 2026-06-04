from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings

from sentence_transformers import SentenceTransformer


DB_PATH = "vectorstore"
COLLECTION_NAME = "interstellar_script"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class MiniLMEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def embed_documents(self, texts):
        return self.model.encode(
            texts,
            normalize_embeddings=True
        ).tolist()

    def embed_query(self, text):
        return self.model.encode(
            text,
            normalize_embeddings=True
        ).tolist()


def format_docs(docs):
    parts = []

    for doc in docs:
        page = doc.metadata.get("page", "unknown")
        text = doc.page_content

        parts.append(f"[Source: Interstellar, page {page}]\n{text}")

    return "\n\n".join(parts)


embedding_function = MiniLMEmbeddings()

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=DB_PATH,
    embedding_function=embedding_function,
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.4,
    num_predict=120,
)

prompt = ChatPromptTemplate.from_template("""
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
""")

def has_good_context(docs):
    return len(docs) > 0

chain = prompt | llm | StrOutputParser()


if __name__ == "__main__":
    question = "tars, what is tesseract?"

    docs = retriever.invoke(question)
    context = format_docs(docs)

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    print("=" * 80)
    print("QUESTION:")
    print(question)

    print("=" * 80)
    print("ANSWER:")
    print(answer)