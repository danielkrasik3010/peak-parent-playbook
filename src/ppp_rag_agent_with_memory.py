"""
Persistent Summarization RAG Agent
----------------------------------

This module implements a Retrieval-Augmented Generation (RAG) agent with persistent
conversation memory and summarization. It combines three key components:

1. **RAG (Retrieval-Augmented Generation):**
   Retrieves relevant documents from a vector database to enrich LLM responses.

2. **Persistent Memory:**
   Stores chat history in an SQL-backed message history, enabling conversations
   to persist across sessions.

3. **Summarization:**
   Automatically summarizes older parts of the conversation to keep context
   manageable while maintaining continuity.

The agent supports multiple LLM backends:
- OpenAI GPT models
- Groq LLaMA models

Typical use case:
-----------------
- Run the script.
- Choose an LLM backend ("OpenAI" or "Groq").
- Provide a session name to persist conversation context across runs.
- Interact with the assistant in a loop where it retrieves relevant docs,
  summarizes prior history, and responds intelligently.
"""

import os
import logging
from datetime import datetime
from dotenv import load_dotenv

# Internal project utilities
from src.utils import load_yaml
from src.prompt_builder_final import build_prompt_from_config
from src.paths import APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH, OUTPUTS_DIR, CHAT_HISTORY_MEMORY_DB_PATH
from src.create_and_ingest_vector_db import get_collection, embed_texts

# LangChain imports
from langchain_community.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory

# Logger setup
logger = logging.getLogger(__name__)

# -------------------------
# Logging Setup
# -------------------------
def setup_logging() -> None:
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(OUTPUTS_DIR, "rag_assistant.log"))
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# -------------------------
# Environment Setup
# -------------------------
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Connect to vector database
collection = get_collection(collection="articles")

# -------------------------
# Document Retrieval
# -------------------------
def retrieve_relevant_documents(query: str, n_results: int = 5, threshold: float = 0.3) -> list[str]:
    logging.info(f"Retrieving relevant documents for query: {query}")
    relevant_results = {"ids": [], "documents": [], "distances": []}

    query_embedding = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances"],
    )

    # Filter by distance threshold
    keep_item = [dist < threshold for dist in results["distances"][0]]
    for i, keep in enumerate(keep_item):
        if keep:
            relevant_results["ids"].append(results["ids"][0][i])
            relevant_results["documents"].append(results["documents"][0][i])
            relevant_results["distances"].append(results["distances"][0][i])

    logging.info("Cosine distances of retrieved docs: " + str(relevant_results["distances"]))
    return relevant_results["documents"]

# -------------------------
# Persistent Summarization RAG Agent
# -------------------------
class PersistentSummarizationRAGAgent:
    def __init__(self, session_name: str, llm_choice: str, prompt_config: dict):
        self.session_name = session_name
        self.prompt_config = prompt_config

        # Setup persistent memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=SQLChatMessageHistory(
                connection=f"sqlite:///{CHAT_HISTORY_MEMORY_DB_PATH}",
                session_id=session_name
            ),
            return_messages=True
        )
        vars = self.memory.load_memory_variables({})
        self.history = vars.get("chat_history", [])

        # Initialize LLM
        if llm_choice.lower() == "openai":
            self.llm = ChatOpenAI(
                model_name="gpt-4",
                temperature=0,
                openai_api_key=OPENAI_API_KEY
            )
        elif llm_choice.lower() == "groq":
            self.llm = ChatGroq(
                model="llama-3.1-8b-instant",
                api_key=GROQ_API_KEY
            )
        else:
            raise ValueError("Invalid LLM choice. Must be 'openai' or 'groq'.")

    # -------------------------
    # Summarize Memory
    # -------------------------
    def summarize_history(self, max_recent: int = 6) -> list:
        recent_messages = self.history[-max_recent:] if len(self.history) > max_recent else self.history
        older_messages = self.history[:-max_recent] if len(self.history) > max_recent else []

        if not older_messages:
            return recent_messages

        older_text = ""
        for msg in older_messages:
            if isinstance(msg, HumanMessage):
                older_text += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                older_text += f"Assistant: {msg.content}\n"

        summary_prompt = f"Summarize this conversation concisely, under 200 words:\n{older_text}"
        try:
            summary_response = self.llm.invoke([HumanMessage(content=summary_prompt)])
            summary_msg = SystemMessage(content=f"Summary of earlier conversation: {summary_response.content}")
            return [summary_msg] + recent_messages
        except Exception as e:
            logger.warning(f"Summarization failed: {e}. Falling back to recent messages only.")
            return recent_messages

    # -------------------------
    # Main Query
    # -------------------------
    def ask(self, user_input: str, n_results: int = 5, threshold: float = 0.3) -> str:
    # Retrieve documents
        relevant_docs = retrieve_relevant_documents(user_input, n_results=n_results, threshold=threshold)
        logging.info(f"Retrieved {len(relevant_docs)} relevant docs.")

        # Build prompt using prompt_builder_final
        input_data = f"Relevant documents:\n\n{relevant_docs}\n\nUser question:\n\n{user_input}"
        rag_prompt = build_prompt_from_config(self.prompt_config, input_data=input_data)

        # Summarize memory and append current user input
        messages = self.summarize_history()
        messages.append(HumanMessage(content=rag_prompt))

        # ✅ FIX: use .invoke for both OpenAI and Groq
        try:
            response = self.llm.invoke(messages)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "Error: LLM call failed."

        # Save to memory
        self.memory.save_context({"input": user_input}, {"output": response.content})
        self.history.append(HumanMessage(content=user_input))
        self.history.append(AIMessage(content=response.content))

        return response.content

# -------------------------
# Main Script
# -------------------------
if __name__ == "__main__":
    setup_logging()

    app_config = load_yaml(APP_CONFIG_FPATH)
    prompt_config_all = load_yaml(PROMPT_CONFIG_FPATH)
    rag_prompt_config = prompt_config_all["rag_ppp_assistant_prompt"]
    vectordb_params = app_config["vectordb"]

    while True:
        llm_choice = input("Choose LLM for this session ('OpenAI' or 'Groq'): ").strip().lower()
        if llm_choice in ["openai", "groq"]:
            break
        print("Invalid choice. Please select 'OpenAI' or 'Groq'.")

    session_name = input("Enter session name (or press Enter for new): ").strip()
    if not session_name:
        session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    agent = PersistentSummarizationRAGAgent(session_name=session_name, llm_choice=llm_choice, prompt_config=rag_prompt_config)

    while True:
        query = input("\nEnter a question, 'config' to change parameters, or 'exit' to quit: ").strip()
        if query.lower() == "exit":
            exit()
        elif query.lower() == "config":
            threshold = float(input("Enter the retrieval threshold: "))
            n_results = int(input("Enter the Top K value: "))
            vectordb_params = {"threshold": threshold, "n_results": n_results}
            continue
        elif query:
            response = agent.ask(query, **vectordb_params)
            logging.info("-" * 100)
            logging.info(f"LLM response:\n{response}\n")
