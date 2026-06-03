# Tars

A local RAG project inspired by TARS from *Interstellar*.

The system uses an Interstellar screenplay PDF as a local knowledge source.  
If the question is related to the script, it answers from retrieved context.  
If no useful context is found, it switches to free improvisation mode with a dry TARS-like style.

## Stack

- Python
- PyMuPDF for PDF text extraction
- SentenceTransformers for embeddings
- ChromaDB for local vector storage
- Ollama with Llama 3.2 3B for local generation

## Pipeline

```text
PDF
→ text extraction
→ chunking with overlap
→ embeddings
→ ChromaDB
→ retrieval
→ Ollama / Llama 3.2 3B
→ answer