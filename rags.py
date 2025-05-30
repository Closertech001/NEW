import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from openai import OpenAI
import re
from symspellpy.symspellpy import SymSpell
import pkg_resources
import time

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="Crescent Chatbot", layout="centered")

# Load Q&A dataset
with open("qa_dataset.json", "r") as f:
    data = json.load(f)

# Extract questions and answers
questions = [entry["question"] for entry in data]
answers = [entry["answer"] for entry in data]

# Load SentenceTransformer and build FAISS index
embedder = SentenceTransformer('all-MiniLM-L6-v2')
question_embeddings = embedder.encode(questions, convert_to_numpy=True)
dimension = question_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(question_embeddings)
index.add(question_embeddings)

# Spell correction
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Abbreviations and synonyms
abbreviations = {
    "u": "you", "r": "are", "ur": "your", "cn": "can", "cud": "could", "shud": "should", "wud": "would",
    "abt": "about", "bcz": "because", "plz": "please", "pls": "please", "tmrw": "tomorrow", "wat": "what",
    "wats": "what is", "info": "information", "yr": "year", "sem": "semester", "admsn": "admission",
    "clg": "college", "sch": "school", "uni": "university", "cresnt": "crescent", "l": "level",
    "d": "the", "doc": "document", "msg": "message", "idk": "i don't know", "imo": "in my opinion",
    "asap": "as soon as possible", "dept": "department", "reg": "registration", "fee": "fees",
    "pg": "postgraduate", "app": "application", "req": "requirement", "nd": "national diploma",
    "a-level": "advanced level", "alevel": "advanced level", "2nd": "second", "1st": "first",
    "nxt": "next", "prev": "previous", "exp": "experience"
}
pidgin_map = {
    "how far": "how are you", "wetin": "what", "no wahala": "no problem", "abeg": "please",
    "sharp sharp": "quickly", "bros": "brother", "guy": "person", "waka": "walk", "chop": "eat",
    "jollof": "rice dish", "nah": "no", "dey": "is", "yarn": "talk", "gbam": "exactly", "ehn": "yes",
    "waka pass": "walk past", "how you dey": "how are you", "i no sabi": "i don't know",
    "make we go": "let's go", "omo": "child", "dash": "give", "carry go": "continue",
    "owo": "money", "pikin": "child", "see as e be": "look how it is", "no vex": "sorry",
    "sharp": "fast", "jare": "please", "e sure": "it is sure", "you sabi": "you know",
    "abeg make you": "please", "how you see am": "what do you think", "carry come": "bring"
}
abbreviations.update(pidgin_map)

synonym_map = {
    "lecturers": "academic staff", "professors": "academic staff", "teachers": "academic staff",
    "instructors": "academic staff", "tutors": "academic staff", "staff members": "staff",
    "head": "dean", "hod": "head of department", "dept": "department", "school": "university",
    "college": "faculty", "course": "subject", "class": "course", "subject": "course",
    "unit": "credit", "credit unit": "unit", "course load": "unit", "non teaching": "non-academic",
    "nonteaching": "non-academic", "admin worker": "non-academic staff",
    "support staff": "non-academic staff", "clerk": "non-academic staff",
    "receptionist": "non-academic staff", "secretary": "non-academic staff",
    "office staff": "non-academic staff", "tech staff": "technical staff",
    "it people": "technical staff", "lab helper": "technical staff",
    "computer staff": "technical staff", "equipment handler": "technical staff",
    "it guy": "technical staff", "hostel": "accommodation", "lodging": "accommodation",
    "room": "accommodation", "school fees": "tuition", "acceptance fee": "admission fee",
    "fees": "tuition", "enrol": "apply", "join": "apply", "sign up": "apply", "admit": "apply",
    "requirement": "criteria", "conditions": "criteria", "needed": "required",
    "needed for": "required for", "who handles": "who manages",
    "who takes care of": "who manages", "computer sci": "computer science",
    "cs": "computer science", "eco": "economics", "stat": "statistics",
    "phy": "physics", "bio": "biology", "chem": "chemistry",
    "mass comm": "mass communication", "comm": "communication", "archi": "architecture",
    "exam": "examination", "tests": "assessments", "marks": "grades"
}
abbreviations.update(synonym_map)

def normalize_text(text):
    text = text.lower()
    for abbr, full in abbreviations.items():
        text = re.sub(rf'\b{re.escape(abbr)}\b', full, text)
    suggest = sym_spell.lookup_compound(text, max_edit_distance=2)
    return suggest[0].term if suggest else text

def render_message(message, is_user=True):
    bg_color = "#DCF8C6" if is_user else "#E1E1E1"
    align = "right" if is_user else "left"
    margin = "10px 0 10px 50px" if is_user else "10px 50px 10px 0"
    animation_class = "slideInRight" if is_user else "slideInLeft"
    return f"""
    <div class="message {animation_class}" style="
        background-color: {bg_color};
        padding: 10px;
        border-radius: 10px;
        max-width: 70%;
        margin: {margin};
        text-align: left;
        float: {align};
        clear: both;
        font-family: Arial, sans-serif;
        font-size: 16px;">
        {message}
    </div><div style="clear: both;"></div>
    """

st.markdown(
    """
    <style>
    .message {
        opacity: 0;
        animation-fill-mode: forwards;
        animation-duration: 0.5s;
        animation-timing-function: ease-out;
    }
    .slideInRight {
        animation-name: slideInRight;
    }
    .slideInLeft {
        animation-name: slideInLeft;
    }
    @keyframes slideInRight {
        from {
            transform: translateX(50%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    @keyframes slideInLeft {
        from {
            transform: translateX(-50%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful university assistant chatbot."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "input_box" not in st.session_state:
    st.session_state["input_box"] = ""

if len(st.session_state.messages) == 0:
    greeting = "Hello! I’m Crescent University Chatbot. How can I assist you today?"
    st.session_state.messages.append({"content": greeting, "is_user": False})

st.title("Crescent University Chatbot")

def get_answer(user_question):
    normalized = normalize_text(user_question)
    query_emb = embedder.encode([normalized], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)
    D, I = index.search(query_emb, k=3)
    threshold = 0.65
    for idx, score in zip(I[0], D[0]):
        if score >= threshold:
            return answers[idx]
    return openai_fallback_response(user_question)

def main():
    for msg in st.session_state.messages:
        st.markdown(render_message(msg["content"], msg["is_user"]), unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your message here...", key="input_box")
        submit = st.form_submit_button("Send")

        if submit and user_input.strip():
            st.session_state.messages.append({"content": user_input.strip(), "is_user": True})
            with st.spinner("Bot is typing..."):
                bot_response = get_answer(user_input.strip())
                time.sleep(0.8)
            st.session_state.messages.append({"content": bot_response, "is_user": False})
            st.session_state["input_box"] = ""
            st.experimental_rerun()

if __name__ == "__main__":
    main()
