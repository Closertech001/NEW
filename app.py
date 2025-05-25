import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import re
import json
import openai
import os
import random
from symspellpy.symspellpy import SymSpell, Verbosity
import pkg_resources

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load SymSpell for spell correction
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Abbreviations mapping
abbreviations = {
    "u": "you", "r": "are", "ur": "your", "ow": "how", "pls": "please", "plz": "please",
    "tmrw": "tomorrow", "cn": "can", "wat": "what", "cud": "could", "shud": "should",
    "wud": "would", "abt": "about", "bcz": "because", "bcoz": "because", "btw": "between",
    "asap": "as soon as possible", "idk": "i don't know", "imo": "in my opinion",
    "msg": "message", "doc": "document", "d": "the", "yr": "year", "sem": "semester",
    "dept": "department", "admsn": "admission", "cresnt": "crescent", "uni": "university",
    "clg": "college", "sch": "school", "info": "information", "l": "level", "CSC": "Computer Science",
    "ECO": "Economics with Operations Research", "PHY": "Physics", "STAT": "Statistics"
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

# Cached resources and data loading
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_data():
    with open("qa_dataset.json", "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
    return pd.DataFrame(qa_pairs)

@st.cache_data
def compute_embeddings(questions):
    model = load_model()
    # preprocess questions before embedding for consistency
    questions_preprocessed = [preprocess_text(q) for q in questions]
    return model.encode(questions_preprocessed, convert_to_tensor=True)

def fallback_openai(user_input, context_qa=None):
    system_prompt = (
        "You are a helpful assistant specialized in Crescent University information. "
        "If you don't know an answer, politely say so and refer to university resources."
    )
    messages = [{"role": "system", "content": system_prompt}]
    if context_qa:
        context_text = f"Here is some relevant university information:\nQ: {context_qa['question']}\nA: {context_qa['answer']}\n\n"
        user_message = context_text + "Answer this question: " + user_input
    else:
        user_message = user_input
    messages.append({"role": "user", "content": user_message})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3
        )
        return response.choices[0].message["content"].strip()
    except Exception:
        return "Sorry, I couldn't reach the server. Try again later."

def find_response(user_input, dataset, embeddings, threshold=0.4):
    model = load_model()
    # preprocess user input for embedding
    user_input_clean = preprocess_text(user_input)
    user_embedding = model.encode(user_input_clean, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(user_embedding, embeddings)[0]
    top_scores, top_indices = torch.topk(cos_scores, k=5)

    top_score = top_scores[0].item()
    top_index = top_indices[0].item()

    if top_score < threshold:
        context_qa = {
            "question": dataset.iloc[top_index]["question"],
            "answer": dataset.iloc[top_index]["answer"]
        }
        gpt_reply = fallback_openai(user_input, context_qa)
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
        response = random.choice(["I think ", "Maybe: ", "Possibly: "]) + response

    return response, department, top_score, related_questions


# --- Streamlit UI setup ---
st.set_page_config(page_title="🎓 Crescent University Chatbot", page_icon="🎓")

# Custom CSS styling for chat
st.markdown("""
<style>
.chat-message-user {
    background-color: #d1eaff;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 10px;
    font-weight: 550;
    color: #000;
    max-width: 70%;
    align-self: flex-end;
}
.chat-message-assistant {
    background-color: #e2e3e5;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 10px;
    font-weight: 600;
    color: #000;
    max-width: 70%;
    align-self: flex-start;
}
.sidebar .stButton>button {
    background-color: #4caf50;
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("🎓 Crescent University Chatbot")

dataset = load_data()
questions = dataset['question'].tolist()
embeddings = compute_embeddings(questions)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "related_questions" not in st.session_state:
    st.session_state.related_questions = []

# Display chat messages with styles
for msg in st.session_state.chat_history:
    role_class = "chat-message-user" if msg["role"] == "user" else "chat-message-assistant"
    with st.chat_message(msg["role"]):
        st.markdown(f'<div class="{role_class}">{msg["content"].replace("\n", "<br>")}</div>', unsafe_allow_html=True)

# User input box
user_input = st.chat_input("Ask me anything about Crescent University...")

if user_input:
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Find bot response
    response, department, confidence, related_questions = find_response(user_input, dataset, embeddings)

    # Append department info if available
    if department:
        response += f"\n\n📘 Department: **{department}**"

    # Add bot message
    st.session_state.chat_history.append({"role": "bot", "content": response})

    # Update related questions for new question
    st.session_state.related_questions = related_questions

    st.experimental_rerun()

# Show related questions buttons below last bot message if any
if st.session_state.related_questions:
    st.markdown("### 💡 Related questions you can ask:")
    cols = st.columns(min(4, len(st.session_state.related_questions)))
    for i, question in enumerate(st.session_state.related_questions[:4]):
        if cols[i].button(question, key=f"related_{i}"):
            # Add related question as new user message
            st.session_state.chat_history.append({"role": "user", "content": question})
            ans, dept, conf, new_related = find_response(question, dataset, embeddings)
            if dept:
                ans += f"\n\n📘 Department: **{dept}**"
            st.session_state.chat_history.append({"role": "bot", "content": ans})

            # Update related questions for next turn
            st.session_state.related_questions = new_related

            st.experimental_rerun()
