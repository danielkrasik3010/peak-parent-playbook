"""
ppp_rag_agent.py

Core module for the Peak Parent Playbook (PPP) RAG-based AI assistant.

This script handles:
- Connecting to the ChromaDB vector store containing embedded professional articles
- Retrieving contextually relevant documents using semantic embeddings
- Building prompts based on configuration and retrieved context
- Generating responses via large language models (OpenAI GPT or Groq)
- Logging queries, retrieved documents, and LLM responses

Functions:
- setup_logging: Configure console and file logging
- retrieve_relevant_documents: Fetch relevant documents from the vector store based on a query
- respond_to_query: Generate a grounded response using retrieved documents and the selected LLM

The module can also be run interactively from the command line to query the assistant,
choose LLM backends, and adjust retrieval parameters.

Author: Daniel Krasik
"""

import os
import logging
from dotenv import load_dotenv
from src.utils import load_yaml
from src.prompt_builder_final import build_prompt_from_config
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage  
from langchain_groq import ChatGroq
from src.paths import APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH, OUTPUTS_DIR
from src.create_and_ingest_vector_db import get_collection, embed_texts

logger = logging.getLogger()

def setup_logging():
    logger.setLevel(logging.INFO)
    # File handler
    file_handler = logging.FileHandler(os.path.join(OUTPUTS_DIR, "rag_assistant.log"))
    file_handler.setLevel(logging.INFO)
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Load environment variables
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Connect to vector DB
collection = get_collection(collection="articles")

def retrieve_relevant_documents(query: str, n_results: int = 5, threshold: float = 0.3) -> list[str]:
    logging.info(f"Retrieving relevant documents for query: {query}")
    relevant_results = {"ids": [], "documents": [], "distances": []}

    # Embed the query
    query_embedding = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances"],
    )

    # Filter results based on cosine distance threshold
    keep_item = [dist < threshold for dist in results["distances"][0]]

    for i, keep in enumerate(keep_item):
        if keep:
            relevant_results["ids"].append(results["ids"][0][i])
            relevant_results["documents"].append(results["documents"][0][i])
            relevant_results["distances"].append(results["distances"][0][i])

    logging.info("Cosine distances of retrieved docs: " + str(relevant_results["distances"]))
    return relevant_results["documents"]

def respond_to_query(prompt_config: dict, query: str, llm, n_results: int = 5, threshold: float = 0.3) -> str:
    relevant_documents = retrieve_relevant_documents(query, n_results=n_results, threshold=threshold)

    logging.info("-" * 100)
    logging.info("Relevant documents:")
    for doc in relevant_documents:
        logging.info(doc)
        logging.info("-" * 100)

    input_data = f"Relevant documents:\n\n{relevant_documents}\n\nUser's question:\n\n{query}"
    rag_assistant_prompt = build_prompt_from_config(prompt_config, input_data=input_data)
    logging.info(f"RAG assistant prompt: {rag_assistant_prompt}")

    # Call the chosen LLM
    if isinstance(llm, ChatOpenAI):
        response = llm([HumanMessage(content=rag_assistant_prompt)])
        return response.content  
    elif isinstance(llm, ChatGroq):
        response = llm.invoke(rag_assistant_prompt)
        return response.content
    else:
        return "Error: Invalid LLM instance."

# Main function to run the assistant interactively
if __name__ == "__main__":
    setup_logging()
    app_config = load_yaml(APP_CONFIG_FPATH)
    prompt_config = load_yaml(PROMPT_CONFIG_FPATH)
    rag_assistant_prompt_config = prompt_config["rag_ppp_assistant_prompt"]
    vectordb_params = app_config["vectordb"]

    # User chooses LLM at session start
    while True:
        llm_choice = input("Choose LLM for this session ('OpenAI' or 'Groq'): ").strip().lower()
        if llm_choice == "openai":
            llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)
            break
        elif llm_choice == "groq":
            if not GROQ_API_KEY:
                print("Error: GROQ_API_KEY is missing in the environment.")
                continue
            llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
            break
        else:
            print("Invalid choice. Please select 'OpenAI' or 'Groq'.")

    # Main query loop
    exit_app = False
    while not exit_app:
        query = input("Enter a question, 'config' to change parameters, or 'exit' to quit: ").strip()
        if query.lower() == "exit":
            exit_app = True
            exit()
        elif query.lower() == "config":
            threshold = float(input("Enter the retrieval threshold: "))
            n_results = int(input("Enter the Top K value: "))
            vectordb_params = {"threshold": threshold, "n_results": n_results}
            continue

        response = respond_to_query(
            prompt_config=rag_assistant_prompt_config,
            query=query,
            llm=llm,
            **vectordb_params,
        )
        logging.info("-" * 100)
        logging.info("LLM response:")
        logging.info(response + "\n\n")