# Peak Parent Playbook (PPP) — RAG Chatbot for Parents of Young Athletes

**An AI-powered, evidence-based coaching assistant for parents**

The Peak Parent Playbook (PPP) is a Retrieval-Augmented Generation (RAG) chatbot designed to guide parents of young athletes in training, nutrition, recovery, sports psychology, and injury prevention. All answers are grounded in trusted publications and PDF guides, providing actionable, kid-safe advice.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline & Architecture](#pipeline--architecture)
- [Core Modules](#core-modules)
- [Prompt Design & Configuration](#prompt-design--configuration)
- [User Interface](#user-interface)
- [Repository Structure](#repository-structure)
- [Quickstart](#quickstart)
- [Demo Queries](#demo-queries)
- [Implementation Notes](#implementation-notes)
- [Extending the Project](#extending-the-project)
- [Development Tips](#development-tips)
- [Contact & License](#contact--license)

---

## Overview

The PPP RAG chatbot transforms trusted professional content into actionable guidance for parents.\
 **Key goals include:**

- Retrieving small, focused chunks of authoritative content.
- Using semantic embeddings and vector search to find relevant information.
- Generating friendly, parent-oriented advice with citations.
- Preserving conversation context via short-term and optional long-term memory.

The system ensures that guidance is **evidence-based, safe, and practical**.

---

## Pipeline & Architecture

### Scrape & Ingest

- Fetch HTML articles and PDFs.
- Extract clean text while removing noise (ads, navigation bars, footers).
- Preserve titles, sections, page numbers, URLs, and publication dates.
- Save structured content as Markdown files in a centralized data directory.

### Embeddings & Vector DB

- Embed content using:
  - **Managed**: OpenAI `text-embedding-3-large`
- Device-aware computation (CUDA, MPS, CPU).
- Persist embeddings and metadata into ChromaDB (local) 

### Retrieval & RAG

- User queries are embedded, and top-k relevant chunks are retrieved.
- Optional reranking for answer-aware selection.
- Context is injected into modular prompts to instruct the LLM.

### Pipeline Diagram


      ┌───────────────┐
      │  Scrape & Save│
      │PDF and HTML   │ 
      │ (scrape.py)   │
      └──────┬────────┘
             │
             ▼
      ┌───────────────┐
      │ Building      │
      │   the prompt  │
      │(prompt_builder│
      │_final.py)     │   
      └──────┬────────┘
             │
             ▼
      ┌───────────────┐
      │ Embeddings &  │
      │ Vector DB     │
      │ (create_and_  │
      │  ingest_vector│
      │  _db.py)      │
      └──────┬────────┘
             │
             ▼
      ┌───────────────┐
      │ Retrieval &   │
      │ RAG Agent     │
      │ (ppp_rag_agent│
      │  .py)         │
      └──────┬────────┘
             │
             ▼
      ┌───────────────┐
      │ Streamlit UI  │
      │ (ui_app.py)   │
      └───────────────┘



## Core Modules

### `__init__.py`

- Initializes Python package structure for the PPP RAG chatbot.
- Ensures modules can be imported cleanly across the repository.

---

### `create_and_ingest_vector_db.py`

- Initializes and manages ChromaDB collections.
- Splits text into pages and semantic sub-chunks (~400–800 tokens, 50–100 token overlap).
- Generates embeddings and inserts chunks into the vector store.
- Functions include: `init_vector_store`, `get_collection`, `split_by_pages`, `semantic_sub_chunk`, `embed_texts`, `add_articles`.

---

### `paths.py`

- Defines consistent paths for data, index, outputs, and notebooks.
- Helps maintain portability and reproducibility across environments.

---

### `ppp_rag_agent.py`

- Core RAG agent connecting vector DB, retrieval, and LLMs.
- Retrieves contextually relevant documents for queries.
- Generates grounded, safe, and parent-friendly responses.
- Supports logging of queries, retrieved chunks, and responses.

---

### `prompt_builder_final.py`

- Builds structured prompts from YAML configuration.
- Includes role, instructions, style, examples, output constraints, and reasoning strategies.
- Functions allow previewing and saving prompts as Markdown for documentation.
- Key functions: `build_prompt_from_config`, `print_prompt_preview`, `save_prompt_to_md`, `load_yaml_config`.

---

### `scrape_articles.py`

- Scrapes HTML and downloads PDFs.
- Extracts text content and saves as structured Markdown.
- Handles multiple URLs with automated file type detection.
- Key functions: `scrape_html`, `scrape_pdf`, `save_markdown`, `scrape_and_save_articles`.

---

### `test_vector_db.py`

- Unit tests for the vector DB functionality.
- Verifies embedding generation, insertion, and retrieval accuracy.

---

### `utils.py`

- Shared utility functions for logging, device detection, file management.
- Ensures robust, maintainable, and reusable code across modules.

---

## Prompt Design & Configuration

Prompts are **modular and YAML-driven**:

**Example of the start of the prompt:**

You are an expert AI assistant that helps parents support their child athletes by answering questions using relevant retrieved documents that you have. 
You specialize in three domains:
1. Strength Training Exercises
2. Nutrition and Dieting
3. How to Be a Supportive Parent


---

## User Interface

- Interactive chat interface via Streamlit.
- Sidebar controls for:
  - LLM selection
  - Retrieval thresholds
  - Top-K results
- Semantic search over curated articles.
- Real-time conversation with color-coded messages.
---

## Repository Structure

Repository Structure
Fitness_Agent/
├── data/
├── outputs/
│   ├── examples/
│   │   └── examples.txt
│   └── vector_db/
│       └── 3d245e97-e72f-4a47-a763-448eba5ad8da/
│   └── chat_history.db
│   └── rag_assistant.log
│   └── rag_PPP_prompt_prompt.md
│   └── test_vector_db.txt
├── src/
│   ├── pycache
│   ├── config/
│   │   ├── config.yaml
│   │   └── prompt_config.yaml 
│   ├── __init__.py
│   ├── create_and_ingest_vector_db.py
│   ├── paths.py
│   ├── ppp_rag_agent.py
│   ├── prompt_builder_final.py
│   ├── test_vector_db.py
│   ├── scrape.py
│   ├── utils. py
├── UI/
│    └── ui_app.py
├── venv/
├── .env
├── .env.example
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md




## Quickstart

**Clone & install dependencies:**


git clone https://github.com/danielkrasik3010/peak-parent-playbook
cd Fitness_Agent
python -m venv venv
# Activate the virtual environment
# macOS / Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate
pip install -r requirements.txt

**Configure .env:**

OPENAI_API_KEY=sk-...
GROQ_API_KEY=...


**Scrape articles:**

python src/scrape_articles.py

**Create the prompt:**
python src/prompt_builder_final.py

**Create and ingest vector db:**

python src/create_and_ingest_vector_db.py 

**Run and test the agent inside vs:**
python src/ppp_rag_agent.py 

**Run chat interface:**
streamlit run UI/ui_app.py

# Demo Queries
"what my child need to eat?"

"does my child needs to take supplements?"

# You can look at bthe examples.txt file to see the responses of the LLM to the Users questions

# Contact & License
Contact & License
Author: Daniel Krasik
Email: daniel.krasik3010@gmail.com
License: MIT License

# Credits
1. Ready Tensor Course
2. StLouisChildrens.org/YoungAthlete
3. Sports Dietitians Australia 
4. https://youthsports.rutgers.edu 


