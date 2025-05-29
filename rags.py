import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import openai
import os

# Set OpenAI key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load dataset
with open("qa_dataset.json", "r") as f:
    data = json.load(f)

# Synonym map
synonym_map = {
    "lecturers": "academic staff", "professors": "academic staff", "teachers": "academic staff",
    "instructors": "academic staff", "tutors": "academic staff",
    "head": "dean", "hod": "dean", "h.o.d": "dean",
    "course": "subject", "class": "course", "courses": "subjects", "classes": "courses",
    "school": "university", "campus": "university", "institution": "university",
    "tech staff": "technical staff", "it people": "technical staff", "lab helper": "technical staff",
    "computer staff": "technical staff", "equipment handler": "technical staff", "it guy": "technical staff",
    "office staff": "non-academic staff", "admin worker": "non-academic staff",
    "support staff": "non-academic staff", "clerk": "non-academic staff", "receptionist": "non-academic staff",
    "school worker": "non-academic staff", "secretary": "non-academic staff",
    "dept": "department", "faculty": "college", "program": "course",
    "physio": "physiology", "cuab": "crescent university", "crescent": "crescent university"
}

# Normalize input
def normalize_text(text):
    text = text.lower()
    for key, value in synonym_map.items():
        text = text.replace(key, value)
    return text

# Embed model
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Preprocess and index questions
questions = [normalize_text(qa["question"]) for qa in data]
embeddings = model.encode(questions, show_progress_bar=False)
index = faiss.IndexFlatL2(embeddings[0].shape[0])
index.add(np.array(embeddings))

# Simple UI message box
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

# RAG fallback
@st.cache_data(show_spinner=False)
def rag_fallback(query):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for Crescent University students."},
                {"role": "user", "content": query}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Sorry, I'm unable to get an answer right now."

# Handle special small talk
def handle_small_talk(msg):
    small_talk = {
        "hi": "Hello! How can I assist you today?",
        "hello": "Hi there! How can I help?",
        "thanks": "You're welcome!",
        "thank you": "You're welcome!",
        "bye": "Goodbye! Have a great day!",
    }
    return small_talk.get(msg.lower())

# Streamlit app layout
st.title("🎓 Crescent University Chatbot")
st.markdown("Ask me anything about Crescent University, Abeokuta!")

if "history" not in st.session_state:
    st.session_state.history = []

# User input
user_input = st.text_input("You:", placeholder="Type your question here...", key="input")

if user_input:
    user_input_clean = normalize_text(user_input)
    st.session_state.history.append((user_input, True))

    small_response = handle_small_talk(user_input_clean)
    if small_response:
        st.session_state.history.append((small_response, False))
    else:
        query_vec = model.encode([user_input_clean])
        D, I = index.search(np.array(query_vec), k=1)
        score = D[0][0]
        match_idx = I[0][0]

        if score < 1.0:  # Good match
            response = data[match_idx]["answer"]
        else:
            response = rag_fallback(user_input_clean)

        st.session_state.history.append((response, False))

# Display conversation
for msg, is_user in st.session_state.history:
    st.markdown(render_message(msg, is_user), unsafe_allow_html=True)

# Footer
st.markdown("<hr style='margin-top:2em;'>", unsafe_allow_html=True)
