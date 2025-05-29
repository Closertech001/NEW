import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os

# Caching model to avoid reloading on every run
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Caching FAISS index and dataset
@st.cache_resource(show_spinner="Loading FAISS index...")
def load_faiss_index():
    with open("qa_dataset.json", "r") as f:
        data = json.load(f)
    corpus = [item["question"] for item in data]
    embeddings = model.encode(corpus, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, data

# Expand abbreviations and synonyms
def expand_synonyms(text):
    synonym_map = {
        "cuab": "crescent university abeokuta",
        "vc": "vice chancellor",
        "hod": "head of department",
        "srf": "student registration form",
        "fresher": "new student",
        "finals": "final year student",
        "convocation": "graduation ceremony",
        # Add more as needed
    }
    words = text.lower().split()
    return " ".join([synonym_map.get(word, word) for word in words])

# Chat bubble renderer
def render_message(message, is_user=True):
    bg_color = "#DCF8C6" if is_user else "#E1E1E1"
    align = "right" if is_user else "left"
    margin = "10px 0 10px 50px" if is_user else "10px 50px 10px 0"
    return f"""
    <div style="
        background-color: {bg_color};
        padding: 10px;
        border-radius: 10px;
        max-width: 70%;
        margin: {margin};
        text-align: left;
        float: {align};
        clear: both;
        font-family: Arial, sans-serif;
        font-size: 14px;
        color:#000;
    ">
        {message}
    </div>
    """

# Load model and index
model = load_model()
index, dataset = load_faiss_index()

# App layout
st.set_page_config(page_title="CUAB Chatbot", layout="wide")
st.markdown("<h1 style='text-align: center;'>🎓 CUAB Chatbot</h1>", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat input and output
user_input = st.chat_input("Ask your question about Crescent University...")

if user_input:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    st.markdown(render_message(user_input, is_user=True), unsafe_allow_html=True)

    # Expand and encode query
    query = expand_synonyms(user_input.strip())
    query_vector = model.encode([query])

    # Search FAISS index
    _, idxs = index.search(np.array(query_vector), k=1)
    answer = dataset[idxs[0][0]]['answer']

    # Display bot response
    st.session_state.messages.append({"role": "bot", "content": answer})
    st.markdown(render_message(answer, is_user=False), unsafe_allow_html=True)

# Render past messages
for msg in st.session_state.messages:
    st.markdown(render_message(msg["content"], is_user=(msg["role"] == "user")), unsafe_allow_html=True)
