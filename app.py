import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import random
import re
from symspellpy.symspellpy import SymSpell, Verbosity
import pkg_resources
import json
import openai
import os

# Set your API key (or load from environment)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Abbreviations for normalization
abbreviations = {
    "u": "you", "r": "are", "ur": "your", "ow": "how", "pls": "please", "plz": "please",
    "tmrw": "tomorrow", "cn": "can", "wat": "what", "cud": "could", "shud": "should",
    "wud": "would", "abt": "about", "bcz": "because", "bcoz": "because", "btw": "between",
    "asap": "as soon as possible", "idk": "i don't know", "imo": "in my opinion",
    "msg": "message", "doc": "document", "d": "the", "yr": "year", "sem": "semester",
    "dept": "department", "admsn": "admission", "cresnt": "crescent", "uni": "university",
    "clg": "college", "sch": "school", "info": "information", "l": "level"
}

# Department mapping
department_map = {
    "GST": "General Studies", "MTH": "Mathematics", "PHY": "Physics", "STA": "Statistics",
    "COS": "Computer Science", "CUAB-CSC": "Computer Science", "CSC": "Computer Science",
    "IFT": "Computer Science", "SEN": "Software Engineering", "ENT": "Entrepreneurship",
    "CYB": "Cybersecurity", "ICT": "Information and Communication Technology",
    "DTS": "Data Science", "CUAB-CPS": "Computer Science", "CUAB-ECO": "Economics with Operations Research",
    "ECO": "Economics with Operations Research", "SSC": "Social Sciences", "CUAB-BCO": "Economics with Operations Research",
    "LIB": "Library Studies", "LAW": "Law (BACOLAW)", "GNS": "General Studies", "ENG": "English",
    "SOS": "Sociology", "PIS": "Political Science", "CPS": "Computer Science",
    "LPI": "Law (BACOLAW)", "ICL": "Law (BACOLAW)", "LPB": "Law (BACOLAW)", "TPT": "Law (BACOLAW)",
    "FAC": "Agricultural Sciences", "ANA": "Anatomy", "BIO": "Biological Sciences",
    "CHM": "Chemical Sciences", "CUAB-BCH": "Biochemistry", "CUAB": "Crescent University - General"
}

def normalize_text(text):
    text = re.sub(r'([^a-zA-Z0-9\s])', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    return text

def preprocess_text(text):
    text = normalize_text(text)
    words = text.split()
    expanded = [abbreviations.get(word.lower(), word) for word in words]
    corrected = []
    for word in expanded:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        corrected.append(suggestions[0].term if suggestions else word)
    return ' '.join(corrected)

def extract_prefix(code):
    match = re.match(r"([A-Z\-]+)", code)
    return match.group(1) if match else None

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_data():
    with open("qa_dataset.json", "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
    return pd.DataFrame(qa_pairs)

@st.cache_data
def compute_question_embeddings(questions: list):
    model = load_model()
    return model.encode(questions, convert_to_tensor=True)

def fallback_openai(user_input):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for Crescent University students."},
                {"role": "user", "content": user_input}
            ],
            temperature=0.3
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return "Sorry, I couldn't reach the server. Try again later."

def find_response(user_input, dataset, embeddings, threshold=0.4):
    model = load_model()
    user_input = preprocess_text(user_input)
    greetings = ["hi", "hello", "hey", "hi there", "greetings", "how are you"]
    if user_input.lower() in greetings:
        return random.choice(["Hello!", "Hi there!", "Hey!", "Greetings!"]), None, 1.0, []

    user_embedding = model.encode(user_input, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(user_embedding, embeddings)[0]
    top_scores, top_indices = torch.topk(cos_scores, k=5)

    top_score = top_scores[0].item()
    top_index = top_indices[0].item()

    if top_score < threshold:
        gpt_reply = fallback_openai(user_input)
        return gpt_reply, None, top_score, []

    response = dataset.iloc[top_index]["answer"]
    question = dataset.iloc[top_index]["question"]

    related_questions = [dataset.iloc[i.item()]["question"] for i in top_indices[1:]]

    match = re.search(r"\b([A-Z]{2,}-?\d{3,})\b", question)
    department = None
    if match:
        code = match.group(1)
        prefix = extract_prefix(code)
        department = department_map.get(prefix, "Unknown")

    if random.random() < 0.2:
        uncertainty = random.choice(["I think ", "Maybe: ", "Possibly: ", "Here's what I found: "])
        response = uncertainty + response

    return response, department, top_score, related_questions

# --- Streamlit UI ---
st.set_page_config(page_title="🎓 Crescent University Chatbot", page_icon="🎓", layout="wide")

# Sidebar styling and content
with st.sidebar:
    st.markdown("<h2 style='color: #2E86C1;'>🎓 Crescent University Chatbot</h2>", unsafe_allow_html=True)
    st.write("Ask me anything about Crescent University.")
    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        st.experimental_rerun()

model = load_model()
dataset = load_data()
question_list = dataset['question'].tolist()
question_embeddings = compute_question_embeddings(question_list)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "prefill_question" in st.session_state:
    prompt = st.session_state.pop("prefill_question")
else:
    prompt = st.chat_input("Ask me anything about Crescent University...")

# Show chat history with colored bubbles
def render_message(role, content):
    if role == "user":
        st.markdown(
            f"""
            <div style='background-color:#D1E8FF; padding:10px; border-radius:10px; max-width:70%; margin-bottom:8px;'>
                <strong>You:</strong><br>{content}
            </div>
            """, unsafe_allow_html=True)
    else:  # assistant
        st.markdown(
            f"""
            <div style='background-color:#F0F0F0; padding:10px; border-radius:10px; max-width:70%; margin-bottom:8px;'>
                <strong>Assistant:</strong><br>{content}
            </div>
            """, unsafe_allow_html=True)

# Display previous messages
for message in st.session_state.chat_history:
    render_message(message["role"], message["content"])

# Process new user input
if prompt is not None and prompt.strip() != "":
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    render_message("user", prompt)

    response, department, confidence, related = find_response(prompt, dataset, question_embeddings)

    response_text = response
    if department:
        response_text += f"<br><em>📘 Department: <strong>{department}</strong></em>"

    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
    render_message("assistant", response_text)

    # Show related questions as buttons
    if related:
        st.markdown("<hr><b>💡 Related questions you might want to ask:</b>", unsafe_allow_html=True)

        # Limit number of related questions shown
        related_subset = random.sample(related, k=min(3, len(related)))

        cols = st.columns(len(related_subset))
        for i, rq in enumerate(related_subset):
            if cols[i].button(rq):
                # When clicked, add question and response to chat history and rerun
                st.session_state.chat_history.append({"role": "user", "content": rq})
                resp2, dep2, conf2, rel2 = find_response(rq, dataset, question_embeddings)
                resp2_text = resp2
                if dep2:
                    resp2_text += f"<br><em>📘 Department: <strong>{dep2}</strong></em>"
                st.session_state.chat_history.append({"role": "assistant", "content": resp2_text})
                st.experimental_rerun()

# Keep chat history manageable
if len(st.session_state.chat_history) > 50:
    st.session_state.chat_history = st.session_state.chat_history[-50:]
