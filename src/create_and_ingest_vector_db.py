"""
create_and_ingest_vector_db.py

Module for initializing, managing, and populating a ChromaDB vector store
with professional articles for the Peak Parent Playbook (PPP) AI assistant.

This script provides utilities for:
- Creating or retrieving a persistent ChromaDB collection
- Splitting text documents into pages and semantically meaningful chunks
- Generating vector embeddings using OpenAI's text-embedding-3-large model
- Adding articles with embeddings into the vector store for RAG-based retrieval

Functions:
- init_vector_store: Initialize or reset a ChromaDB collection
- get_collection: Retrieve an existing ChromaDB collection
- split_by_pages: Split text into pages based on page markers
- semantic_sub_chunk: Further split long pages into semantic chunks
- embed_texts: Convert text chunks into embeddings
- add_articles: Insert multiple articles into the vector store
- main: Load all articles and populate the vector store

Author: Daniel Krasik
"""

import os
import shutil
import chromadb
from pathlib import Path
from typing import List
from src.paths import VECTOR_DB_DIR
from src.utils import read_all_articles, ensure_env
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re




def init_vector_store(
    db_path: Path = Path(VECTOR_DB_DIR),
    collection: str = "articles",  
    reset: bool = False,
) -> chromadb.Collection:
    """
    Initialize or retrieve a ChromaDB collection with persistent storage.
    """
    if db_path.exists() and reset:
        shutil.rmtree(db_path)
    db_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_path))

    try:
        return client.get_collection(name=collection)
    except Exception:
        print(f"Creating new collection: {collection}")
        return client.create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine", "hnsw:batch_size": 10_000}
        )


def get_collection(
    db_path: Path = Path(VECTOR_DB_DIR),
    collection: str = "articles"  
) -> chromadb.Collection:
    """Retrieve an existing ChromaDB collection."""
    return chromadb.PersistentClient(path=str(db_path)).get_collection(name=collection)


def split_by_pages(text: str) -> List[str]:
    """
    Split text based on '## Page X' markers.
    """
    pages = re.split(r"(?:## Page \d+)", text)
    pages = [p.strip() for p in pages if p.strip()]  # remove empty strings
    return pages


def semantic_sub_chunk(page_text: str, size: int = 1000, overlap: int = 200) -> List[str]:
    """
    For long pages, further split them using semantic chunking.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.split_text(page_text)


def embed_texts(chunks: List[str]) -> List[List[float]]:
    """
    Convert text chunks into vector embeddings using OpenAI's text-embedding-3-large.
    """
    embedder = OpenAIEmbeddings(model="text-embedding-3-large")
    return embedder.embed_documents(chunks)


def add_articles(collection: chromadb.Collection, docs: List[str]) -> None:
    """
    Insert articles into the vector store with page-level embeddings.
    """
    next_id = collection.count()

    for doc_index, doc in enumerate(docs):
        pages = split_by_pages(doc)

        for page_index, page in enumerate(pages):
            # If the page is long, further split it
            if len(page) > 1200:  
                chunks = semantic_sub_chunk(page)
            else:
                chunks = [page]

            vectors = embed_texts(chunks)
            ids = [
                f"article_{doc_index}_page_{page_index}_chunk_{i}"
                for i in range(len(chunks))
            ]
            collection.add(embeddings=vectors, ids=ids, documents=chunks)
            next_id += len(chunks)


def main():
    # Ensure environment variables (like OPENAI_API_KEY) are loaded
    ensure_env()
    store = init_vector_store(reset=True)
    articles = read_all_articles()  
    add_articles(store, articles)
    print(f"Vector store contains {store.count()} documents.")


if __name__ == "__main__":
    main()
