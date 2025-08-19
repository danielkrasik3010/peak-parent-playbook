from chromadb.utils import embedding_functions
from utils import ensure_env
from paths import VECTOR_DB_DIR, OUTPUTS_DIR
import chromadb
import os
import numpy as np
from pathlib import Path

# Load env so OpenAI works
ensure_env()

# Connect to your collection
client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
collection = client.get_collection(name="articles")

# Grab the first 5 docs with embeddings
all_docs = collection.get(include=["documents", "embeddings"])
# Prepare a sample of documents to check
docs_to_check = {
    "ids": all_docs["ids"][:5],
    "documents": all_docs["documents"][:5],
    "embeddings": all_docs["embeddings"][:5]
}

# Query text for similarity
query_text = "Fitness tips for better health and muscle growth"

# Use same embedding function
embedder = embedding_functions.OpenAIEmbeddingFunction(
    model_name="text-embedding-3-large",
    api_key=os.getenv("OPENAI_API_KEY")
)

query_vector = embedder([query_text])[0]

# Cosine similarity helper
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Prepare output
output_lines = []
for doc_id, doc, vec in zip(docs_to_check["ids"], docs_to_check["documents"], docs_to_check["embeddings"]):
    score = cosine_similarity(query_vector, vec)
    result_str = f"ID: {doc_id}\nDoc:\n{doc}\n\nCosine similarity: {score:.4f}\n{'-'*60}"
    print(result_str)
    output_lines.append(result_str)

# Ensure outputs directory exists
Path(OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)

# Save to file inside outputs dir
output_path = Path(OUTPUTS_DIR) / "test_vector_db.txt"
output_path.write_text("\n\n".join(output_lines), encoding="utf-8")
print(f"\nSaved output to {output_path.resolve()}")