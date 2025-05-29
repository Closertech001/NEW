import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import openai
import os
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

# Abbreviation mapping
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

# Synonym map
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
        text = text.replace(abbr, full)
    for key, value in synonym_map.items():
        text = text.replace(key, value)
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    if suggestions:
        text = suggestions[0].term
    return text

# Load model only once, cache_resource because it's a heavy resource and unhashable
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Prepare the questions list outside cached function (simple list of strings)
questions = [normalize_text(qa["question"]) for qa in data]

# Cache index building, but only pass hashable simple input (list of strings)
@st.cache_data(show_spinner=False)
def build_faiss_index(questions):
    embeddings = model.encode(questions, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return embeddings, index

embeddings, index = build_faiss_index(questions)

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
    except Exception:
        return "Sorry, I'm unable to get an answer right now."

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
        font-weight: 600;
        color:#000;
    ">
        {message}
    </div>
    """

# Handle special small talk
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

# Streamlit app layout
st.title("🎓 Crescent University Chatbot")
st.markdown("Ask me anything about Crescent University, Abeokuta!")

if "history" not in st.session_state:
    st.session_state.history = []

# User input
user_input = st.text_input("You:", placeholder="Type your question here...")

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

        if score < 1.0:
            response = data[match_idx]["answer"]
        else:
            with st.spinner("Thinking..."):
                response = rag_fallback(user_input_clean)

        st.session_state.history.append((response, False))

    st.experimental_rerun()

# Display conversation
for msg, is_user in st.session_state.history:
    st.markdown(render_message(msg, is_user), unsafe_allow_html=True)
