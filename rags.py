import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from openai import OpenAI
import re
from symspellpy.symspellpy import SymSpell
import pkg_resources

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="Crescent Chatbot", layout="centered")

# Load Q&A dataset (expects list of dicts with keys: question, answer)
with open("qa_dataset.json", "r") as f:
    data = json.load(f)

# Initialize SymSpell for spell correction
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Precompute embeddings for questions in dataset
questions = [item["question"] for item in data]
question_embeddings = embedder.encode(questions, convert_to_numpy=True)

# Build FAISS index
embedding_dim = question_embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)  # Using inner product for cosine similarity
# Normalize embeddings for cosine similarity
faiss.normalize_L2(question_embeddings)
index.add(question_embeddings)

# Abbreviation, slang and synonym maps (same as before)
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

def normalize_text(text):
    text = text.lower()
    # Replace abbreviations and slang word-by-word
    words = text.split()
    normalized_words = []
    for w in words:
        w = abbreviations.get(w, w)
        w = synonym_map.get(w, w)
        normalized_words.append(w)
    normalized_text = " ".join(normalized_words)
    # Use symspell to correct spelling
    suggest = sym_spell.lookup_compound(normalized_text, max_edit_distance=2)
    return suggest[0].term if suggest else normalized_text

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
    .typing-indicator {
        font-style: italic;
        color: gray;
        margin: 10px 0 10px 50px;
        clear: both;
        font-family: Arial, sans-serif;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def openai_fallback_response(question):
    prompt = f"You are an assistant for Crescent University. Answer this question:\n{question}\n"
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return "Sorry, I couldn't generate an answer at this time."

def chatbot_response(question, threshold=0.6):
    normalized_question = normalize_text(question)
    q_emb = embedder.encode([normalized_question], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)

    D, I = index.search(q_emb, k=3)  # top 3 results

    # D contains cosine similarity scores (since normalized)
    best_score = D[0][0]
    best_idx = I[0][0]

    if best_score >= threshold:
        answer = data[best_idx]["answer"]
        return answer
    else:
        # fallback to GPT
        return openai_fallback_response(question)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "typing" not in st.session_state:
    st.session_state.typing = False
if "greeted" not in st.session_state:
    st.session_state.greeted = False

def main():
    st.title("Crescent University Chatbot")

    with st.sidebar:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.greeted = False
            st.session_state.typing = False
            st.experimental_rerun()

    if not st.session_state.greeted:
        greeting_msg = "Hi there! I am Crescent Chatbot. How can I help you today?"
        st.session_state.messages.append({"role": "bot", "content": greeting_msg})
        st.session_state.greeted = True

    for msg in st.session_state.messages:
        is_user = msg["role"] == "user"
        st.markdown(render_message(msg["content"], is_user), unsafe_allow_html=True)

    if st.session_state.typing:
        st.markdown('<div class="typing-indicator">Bot is typing...</div>', unsafe_allow_html=True)

    user_input = st.text_input("You:", key="input", placeholder="Ask me anything...")

    if user_input and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input.strip()})
        st.session_state.input = ""
        st.session_state.typing = True
        st.experimental_rerun()

    if st.session_state.typing and st.session_state.messages:
        last_msg = st.session_state.messages[-1]
        if last_msg["role"] == "user":
            response = chatbot_response(last_msg["content"])
            st.session_state.messages.append({"role": "bot", "content": response})
            st.session_state.typing = False
            st.experimental_rerun()

if __name__ == "__main__":
    main()
