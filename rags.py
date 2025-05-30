import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from openai import OpenAI
import re
from symspellpy.symspellpy import SymSpell
import pkg_resources
import tiktoken
import logging
import os
import random

# 🔐 OpenAI API client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 🚼 Page setup
st.set_page_config(page_title="Crescent Chatbot", layout="centered")

# 📚 Load data
with open("qa_dataset.json", "r") as f:
    data = json.load(f)
filtered_data = data

# 🧠 SymSpell + slang/abbreviations/synonyms
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dict_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
if os.path.exists("pidgin_dict.txt"):
    sym_spell.load_dictionary("pidgin_dict.txt", term_index=0, count_index=1)

abbreviations = {
    "u": "you", "r": "are", "ur": "your", "cn": "can", "cud": "could", "shud": "should", "wud": "would",
    "abt": "about", "bcz": "because", "plz": "please", "pls": "please", "tmrw": "tomorrow", "wat": "what",
    "wats": "what is", "info": "information", "yr": "year", "sem": "semester", "admsn": "admission",
    "clg": "college", "sch": "school", "uni": "university", "cresnt": "crescent", "l": "level", 
    "d": "the", "msg": "message", "idk": "i don't know", "imo": "in my opinion", "asap": "as soon as possible",
    "dept": "department", "reg": "registration", "fee": "fees", "pg": "postgraduate", "app": "application",
    "req": "requirement", "nd": "national diploma", "a-level": "advanced level", "alevel": "advanced level",
    "2nd": "second", "1st": "first", "nxt": "next", "prev": "previous", "exp": "experience"
}
abbreviations.update({
    "how far": "how are you", "wetin": "what", "no wahala": "no problem", "abeg": "please",
    "sharp sharp": "quickly", "bros": "brother", "guy": "person", "waka": "walk", "chop": "eat",
    "jollof": "rice dish", "nah": "no", "dey": "is", "yarn": "talk", "gbam": "exactly", "ehn": "yes",
    "waka pass": "walk past", "how you dey": "how are you", "i no sabi": "i don't know",
    "make we go": "let's go", "omo": "child", "dash": "give", "carry go": "continue", "owo": "money",
    "pikin": "child", "see as e be": "look how it is", "no vex": "sorry", "sharp": "fast", 
    "jare": "please", "e sure": "it is sure", "you sabi": "you know", "abeg make you": "please", 
    "how you see am": "what do you think", "carry come": "bring"
})

synonym_map = {
    "lecturers": "academic staff", "professors": "academic staff", "teachers": "academic staff",
    "instructors": "academic staff", "tutors": "academic staff", "staff members": "staff",
    "head": "dean", "hod": "head of department", "dept": "department", "school": "university",
    "college": "faculty", "course": "subject", "class": "course", "subject": "course", 
    "unit": "credit", "credit unit": "unit", "course load": "unit", "non teaching": "non-academic",
    "admin worker": "non-academic staff", "support staff": "non-academic staff", "clerk": "non-academic staff",
    "receptionist": "non-academic staff", "secretary": "non-academic staff", "tech staff": "technical staff",
    "hostel": "accommodation", "lodging": "accommodation", "room": "accommodation", "school fees": "tuition",
    "acceptance fee": "admission fee", "fees": "tuition", "enrol": "apply", "join": "apply", 
    "sign up": "apply", "admit": "apply", "requirement": "criteria", "conditions": "criteria",
    "needed": "required", "needed for": "required for", "who handles": "who manages", 
    "cs": "computer science", "eco": "economics", "stat": "statistics", "phy": "physics",
    "bio": "biology", "chem": "chemistry", "mass comm": "mass communication", 
    "archi": "architecture", "exam": "examination", "marks": "grades"
}

def normalize_text(text):
    text = text.lower()
    for abbr, full in abbreviations.items():
        text = re.sub(rf'\b{re.escape(abbr)}\b', full, text)
    for key, val in synonym_map.items():
        text = re.sub(rf'\b{re.escape(key)}\b', val, text)
    suggest = sym_spell.lookup_compound(text, max_edit_distance=2)
    return suggest[0].term if suggest else text

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
model = load_model()

@st.cache_resource
def build_index():
    questions = [normalize_text(q["question"]) for q in filtered_data]
    emb = model.encode(questions, show_progress_bar=False)
    emb = np.array(emb).astype("float32")
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(emb.shape[1]), emb.shape[1], 100)
    index.train(emb)
    index.add(emb)
    return index, emb, questions
index, embeddings, questions = build_index()

def extract_course_code(text):
    match = re.search(r'\b([A-Za-z]{3}\s?\d{3})\b', text)
    if match:
        return match.group(1).replace(" ", "").upper()
    return None

def get_course_info(course_code):
    code_lower = course_code.lower()
    for entry in data:
        if code_lower in entry.get("question", "").lower() or code_lower == entry.get("course_code", "").lower():
            name = entry.get("course_name", "Unknown course")
            level = entry.get("level", "Unknown level")
            return f"{course_code} is '{name}' and it is done at level {level}."
    return f"Sorry, I couldn't find information about {course_code}."

def rag_fallback_with_context(query, top_k_matches):
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
        context_parts, total_tokens = [], 0
        for i in top_k_matches:
            if i < len(filtered_data):
                pair = f"Q: {filtered_data[i]['question']}\nA: {filtered_data[i]['answer']}"
                tokens = len(encoding.encode(pair))
                if total_tokens + tokens > 3596:
                    break
                context_parts.append(pair)
                total_tokens += tokens

        messages = [
            {"role": "system", "content": "You are a helpful assistant using Crescent University's dataset."},
            {"role": "system", "content": f"Context:\n{chr(10).join(context_parts)}"},
            {"role": "user", "content": query}
        ]

        response = client.chat.completions.create(model="gpt-4", messages=messages)
        return response.choices[0].message.content.strip()

    except Exception as e:
        logging.warning(f"OpenAI fallback error: {e}")
        return "I couldn't find an exact match. Could you try rephrasing?"

UNMATCHED_LOG = "unmatched_queries.log"
def log_unmatched_query(query):
    with open(UNMATCHED_LOG, "a") as f:
        f.write(query + "\n")

dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)

if dark_mode:
    st.markdown("""
        <style>
        body, .message { background-color: #121212 !important; color: #ffffff !important; }
        .user-msg { background-color: #1e88e5 !important; color: white !important; }
        .bot-msg { background-color: #2c2c2c !important; color: white !important; }
        </style>
    """, unsafe_allow_html=True)

def is_greeting(text):
    return text.lower().strip() in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]

def is_farewell(text):
    return text.lower().strip() in ["bye", "goodbye", "see you", "later", "farewell", "cya", "peace", "exit"]

def get_random_greeting_response():
    return random.choice([
        "Hello! How can I assist you today?",
        "Hi there! What can I help you with?",
        "Hey! Feel free to ask me anything about Crescent University.",
        "Greetings! How may I be of service?",
        "Hello! Ready to help you with any questions."
    ])

def get_random_farewell_response():
    return random.choice([
        "Goodbye! Have a great day!",
        "See you later! Feel free to come back anytime.",
        "Bye! Take care!",
        "Farewell! Let me know if you need anything else.",
        "Peace out! Hope to chat again soon."
    ])

def render_message(message, is_user=False):
    css_class = "user-msg" if is_user else "bot-msg"
    return f'<div class="message {css_class}">{message}</div>'

st.markdown("""
    <style>
    input[type="text"] {
        padding: 12px;
        border-radius: 12px;
        font-size: 16px;
        border: 1px solid #ccc;
    }
    button[kind="primary"] {
        background-color: #2e7d32 !important;
        color: white !important;
        border-radius: 12px !important;
        font-size: 16px !important;
        padding: 8px 20px !important;
        margin-top: 5px;
    }
    button[kind="primary"]:hover {
        background-color: #1b5e20 !important;
    }
    .message {
        padding: 12px;
        border-radius: 16px;
        margin: 8px 0;
        max-width: 75%;
        word-wrap: break-word;
        font-family: Arial, sans-serif;
        font-size: 16px;
        line-height: 1.5;
    }
    .user-msg {
        background-color: #dcf8c6;
        margin-left: auto;
        text-align: right;
    }
    .bot-msg {
        background-color: #f1f0f0;
        margin-right: auto;
        text-align: left;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    st.title("Crescent University Chatbot")

    if st.sidebar.button("Clear Chat"):
        st.session_state.history = []

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask me anything about Crescent University:")
        submitted = st.form_submit_button("Send")

    if submitted and user_input:
        norm_input = normalize_text(user_input)

        with st.spinner("Bot is typing..."):
            if is_greeting(user_input):
                answer = get_random_greeting_response()
            elif is_farewell(user_input):
                answer = get_random_farewell_response()
            else:
                course_code = extract_course_code(norm_input)
                if course_code:
                    answer = get_course_info(course_code)
                else:
                    query_emb = model.encode([norm_input], show_progress_bar=False)
                    D, I = index.search(np.array(query_emb).astype("float32"), 10)
                    best_score = D[0][0]
                    best_idx = I[0][0]
                    if best_score < 1.0 and best_idx < len(filtered_data):
                        answer = filtered_data[best_idx]["answer"]
                    else:
                        answer = rag_fallback_with_context(user_input, I[0])
                        if "couldn't find" in answer.lower() or "try rephrasing" in answer.lower():
                            log_unmatched_query(user_input)

        st.session_state.history.append(("user", user_input))
        st.session_state.history.append(("bot", answer))

    for role, msg in st.session_state.history:
        st.markdown(render_message(msg, is_user=(role == "user")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
