import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import openai
import os
import re
from symspellpy.symspellpy import SymSpell, Verbosity
import pkg_resources

# Set OpenAI key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load dataset
with open("qa_dataset.json", "r") as f:
    data = json.load(f)

# Initialize SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Abbreviations and Synonym Maps
abbreviations = {
    "u": "you", "r": "are", "ur": "your", "ow": "how", "pls": "please", "plz": "please",
    "tmrw": "tomorrow", "cn": "can", "wat": "what", "cud": "could", "shud": "should",
    "wud": "would", "abt": "about", "bcz": "because", "btw": "between", "asap": "as soon as possible",
    "idk": "i don't know", "imo": "in my opinion", "msg": "message", "doc": "document", "d": "the",
    "yr": "year", "sem": "semester", "dept": "department", "admsn": "admission",
    "cresnt": "crescent", "uni": "university", "clg": "college", "sch": "school",
    "info": "information", "l": "level", "CSC": "Computer Science", "ECO": "Economics with Operations Research",
    "PHY": "Physics", "STAT": "Statistics", "1st": "First", "2nd": "Second",
    "tech staff": "technical staff", "it people": "technical staff", "lab helper": "technical staff",
    "computer staff": "technical staff", "equipment handler": "technical staff", "it guy": "technical staff",
    "office staff": "non-academic staff", "admin worker": "non-academic staff", "support staff": "non-academic staff",
    "clerk": "non-academic staff", "receptionist": "non-academic staff", "school worker": "non-academic staff",
    "secretary": "non-academic staff"
}

synonym_map = {
    "lecturers": "academic staff", "professors": "academic staff", "teachers": "academic staff",
    "instructors": "academic staff", "tutors": "academic staff",
    "head": "dean", "hod": "dean", "h.o.d": "dean",
    "course": "subject", "class": "course", "courses": "subjects", "classes": "courses",
    "school": "university", "campus": "university", "institution": "university",
    "dept": "department", "faculty": "college", "program": "course",
    "physio": "physiology", "cuab": "crescent university", "crescent": "crescent university"
}

# Normalize input
def normalize_text(text):
    text = text.lower()
    for abbr, full in abbreviations.items():
        text = re.sub(rf'\b{re.escape(abbr)}\b', full, text)
    for key, value in synonym_map.items():
        text = re.sub(rf'\b{re.escape(key)}\b', value, text)
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    if suggestions:
        text = suggestions[0].term
    return text

# Load model
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Build FAISS index
@st.cache_resource(show_spinner=False)
def build_faiss_index():
    questions = [normalize_text(qa["question"]) for qa in data]
    embeddings = model.encode(questions, show_progress_bar=False)
    dim = embeddings[0].shape[0]

    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, 100)
    index.train(np.array(embeddings))
    index.add(np.array(embeddings))

    return index, embeddings, questions

index, embeddings, questions = build_faiss_index()

# Message renderer
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

# GPT Fallback
@st.cache_data(show_spinner=False)
def rag_fallback_with_context(query, top_k_matches):
    context_text = "\n".join([f"Q: {data[i]['question']}\nA: {data[i]['answer']}" for i in top_k_matches])
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Faster than GPT-4
            messages=[
                {"role": "system", "content": "You are a helpful assistant using Crescent University's dataset."},
                {"role": "user", "content": f"Refer to the following:\n{context_text}\n\nNow answer this:\n{query}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "Sorry, I'm unable to get an answer right now."

# Handle greetings
def handle_small_talk(msg):
    small_talk = {
        "hi": "Hello! How can I assist you today?",
        "hello": "Hi there! How can I help?",
        "thanks": "You're welcome!",
        "thank you": "You're welcome!",
        "bye": "Goodbye! Have a great day!",
        "goodbye": "Bye! Take care.",
        "how are you": "I'm just a bot, but I'm here to help you!"
    }
    return small_talk.get(msg.lower())

# Streamlit UI
st.title("🎓 Crescent University Chatbot")
st.markdown("Ask me anything about Crescent University, Abeokuta!")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You:", placeholder="Type your question here...")

if user_input:
    user_input_clean = normalize_text(user_input)
    st.session_state.history.append((user_input, True))

    small_response = handle_small_talk(user_input_clean)
    if small_response:
        st.session_state.history.append((small_response, False))
    else:
        query_vec = model.encode([user_input_clean])
        index.nprobe = 10  # Improves accuracy for IndexIVFFlat
        D, I = index.search(np.array(query_vec), k=3)
        scores = D[0]
        indices = I[0]

        if scores[0] < 0.9:
            response = data[indices[0]]["answer"]
        else:
            response = rag_fallback_with_context(user_input_clean, indices)

        st.session_state.history.append((response, False))

# Display chat history
for msg, is_user in st.session_state.history:
    st.markdown(render_message(msg, is_user), unsafe_allow_html=True)

st.markdown("<hr style='margin-top:2em;'>", unsafe_allow_html=True)
st.caption("Built for Crescent University using FAISS + RAG hybrid.")
