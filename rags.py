import streamlit as st
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load your data (example: JSON file with Q&A and metadata)
@st.cache_data
def load_data():
    with open("data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# Initialize model once
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Build FAISS index
@st.cache_resource
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Render chat messages with optional metadata
def render_message(message, is_user=True, abbreviation=None, summary=None, department=None):
    bg_color = "#DCF8C6" if is_user else "#E1E1E1"
    align = "right" if is_user else "left"
    margin = "10px 0 10px 50px" if is_user else "10px 50px 10px 0"

    extra_info_html = ""
    if abbreviation:
        extra_info_html += f'<div style="font-weight:bold; margin-top: 5px;">Abbreviation: {abbreviation}</div>'
    if summary:
        extra_info_html += f'<div style="font-style: italic; margin-top: 5px;">Summary: {summary}</div>'
    if department:
        extra_info_html += f'<div style="color: #555; margin-top: 5px;">Department: {department}</div>'

# Example synonym map
SYNONYM_MAP = {
    "cuab": "crescent university abeokuta",
    "vc": "vice chancellor",
    "hod": "head of department",
    "srf": "student registration form",
    "csc": "computer science",
    # Add more if needed
}

def expand_synonyms(text):
    words = text.lower().split()
    return " ".join([SYNONYM_MAP.get(word, word) for word in words])

    
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
        {extra_info_html}
    </div>
    """

def main():
    st.title("University Q&A Chatbot")

    data = load_data()
    model = load_model()

    # Prepare embeddings array
    questions = [entry["question"] for entry in data]
    embeddings = np.array([model.encode(q) for q in questions])
    index = build_faiss_index(embeddings)

    # Session state for chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    user_input = st.text_input("Ask your question:")

    if user_input:
        # Show user question
        st.session_state.history.append({"message": user_input, "is_user": True})

        # Embed user query and search
        expanded_input = expand_synonyms(user_input)
        query_vec = model.encode(user_input).reshape(1, -1)
        distances, indices = index.search(query_vec, k=1)  # top 1

        best_match = data[indices[0][0]]
        answer = best_match.get("answer", "Sorry, I don't have an answer for that.")
        abbreviation = best_match.get("abbreviation")
        summary = best_match.get("summary")
        department = best_match.get("department")

        # Append bot answer with metadata
        st.session_state.history.append(
            {
                "message": answer,
                "is_user": False,
                "abbreviation": abbreviation,
                "summary": summary,
                "department": department,
            }
        )

    # Render chat history
    for chat in st.session_state.history:
        st.markdown(
            render_message(
                message=chat["message"],
                is_user=chat["is_user"],
                abbreviation=chat.get("abbreviation"),
                summary=chat.get("summary"),
                department=chat.get("department"),
            ),
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
