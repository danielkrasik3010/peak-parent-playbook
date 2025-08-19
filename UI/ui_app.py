import sys
import os
import streamlit as st
from dotenv import load_dotenv
import yaml
import logging

# -------------------------
# Add Fitness_Agent root folder to Python path
# -------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# -------------------------
# Import project modules
# -------------------------
from src.ppp_rag_agent import respond_to_query
from src.create_and_ingest_vector_db import get_collection, embed_texts
from src.paths import APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH
from langchain_community.chat_models import ChatOpenAI
from langchain_groq import ChatGroq

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()

# -------------------------
# Load configurations
# -------------------------
with open(APP_CONFIG_FPATH, "r") as f:
    app_config = yaml.safe_load(f)

with open(PROMPT_CONFIG_FPATH, "r") as f:
    prompt_config_all = yaml.safe_load(f)

rag_prompt_config = prompt_config_all["rag_ppp_assistant_prompt"]

# -------------------------
# Streamlit page setup & theme
# -------------------------
st.set_page_config(
    page_title="Peak Parent Playbook",
    page_icon="🏅",
    layout="wide"
)

# -------------------------
# Styling and visual enhancements
# -------------------------
st.markdown("""
<style>
body {background-color: #f7f9fc; color: #1f2937; font-family: 'Segoe UI', sans-serif;}
h1, h2, h3, h4 {color: #111827;}
.stButton>button {background-color: #3b82f6; color: white; border-radius: 8px; padding: 0.5em 1em;}
.stButton>button:hover {background-color: #2563eb; color: white;}
.stSlider>div {color: #1f2937;}
.chat-message-user {background-color: #ffedd5; border-radius: 12px; padding: 10px; margin-bottom: 6px;}
.chat-message-ppp {background-color: #dbeafe; border-radius: 12px; padding: 10px; margin-bottom: 6px;}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Header section with logo / top image
# -------------------------
# If the image is in the same folder as this script:
st.image(r"C:\Users\97254\Documents\Ready_Tensor_AI_Course\course_workspace\Fitness_Agent\data\father_and_son.jpg", width=250)

# OR if the image is in a "data" subfolder:
# st.image("data/father_and_son.jpg", width=250)

st.markdown(
    """
    <h1>Peak Parent Playbook 🏆</h1>
    <p>Your AI-powered parenting coach for young athletes. Get evidence-based guidance on training, nutrition, and emotional support.</p>
    <hr>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Sidebar - Settings
# -------------------------
st.sidebar.header("Configuration")
st.sidebar.markdown("Adjust the RAG agent retrieval settings:")

llm_choice = st.sidebar.radio(
    "Choose LLM",
    ["OpenAI", "Groq"],
    help="Select the language model to generate responses."
)

threshold = st.sidebar.slider(
    "Retrieval threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Lower values are stricter; higher values retrieve more documents."
)

n_results = st.sidebar.slider(
    "Top K results",
    min_value=1,
    max_value=10,
    value=5,
    help="Number of top relevant documents retrieved for each query."
)

# -------------------------
# Initialize chat state
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# LLM setup
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if llm_choice.lower() == "openai":
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)
else:
    if not GROQ_API_KEY:
        st.sidebar.warning("GROQ API key not found. Using OpenAI as fallback.")
        llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)
    else:
        llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)

# -------------------------
# Connect to vector DB (Chroma)
# -------------------------
collection = get_collection(collection="articles")

# -------------------------
# Document retrieval function
# -------------------------
def retrieve_relevant_documents(query: str, n_results: int = 5, threshold: float = 0.5):
    logging.info(f"Retrieving relevant documents for query: {query}")
    relevant_results = {"documents": [], "distances": []}
    query_embedding = embed_texts([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances"]
    )
    for doc, distance in zip(results["documents"][0], results["distances"][0]):
        if distance < threshold:
            relevant_results["documents"].append(doc)
            relevant_results["distances"].append(distance)
    return relevant_results["documents"], relevant_results["distances"]

# -------------------------
# Chat interface
# -------------------------
st.subheader("💬 Ask your question")
user_input = st.text_input("Type your question here...", placeholder="e.g., What should my child eat before training?")

if st.button("Send") and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner("Retrieving relevant documents..."):
        relevant_docs, distances = retrieve_relevant_documents(user_input, n_results=n_results, threshold=threshold)
    if not relevant_docs:
        response_text = "I am sorry but this question is not answerable given the documents or knowledge that I have."
    else:
        response_text = respond_to_query(
            prompt_config=rag_prompt_config,
            query=user_input,
            llm=llm,
            n_results=n_results,
            threshold=threshold
        )
    st.session_state.messages.append({"role": "ppp", "content": response_text})

# -------------------------
# Display chat history with custom colors
# -------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-message-user'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-message-ppp'><b>Peak Parent Playbook:</b> {msg['content']}</div>", unsafe_allow_html=True)

# -------------------------
# Footer
# -------------------------
st.markdown(
    "<hr><center><small>Developed by Daniel Krasik &nbsp;|&nbsp; Powered by Streamlit & LangChain</small></center>",
    unsafe_allow_html=True
)
