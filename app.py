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

# Load SymSpell for typo correction
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
    "clg": "college", "sch": "school", "info": "information", "l": "level", "csc": "Computer Science", 
    "Mass Comm": "Mass Commuincation", "ACC": "Accounting"
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

# Text normalization and preprocessing
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

# Extract department prefix from question code
def extract_prefix(code):
    match = re.search(r"\b([A-Z]{2,}(?:-[A-Z]{2,})*-\d{3,})\b", question)
    return match.group(1) if match else None

# Load SentenceTransformer model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load Q&A dataset
@st.cache_resource
def load_data():
    with open("qa_dataset.json", "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
    return pd.DataFrame(qa_pairs)

# Compute embeddings for questions
@st.cache_data
def compute_question_embeddings(questions: list):
    model = load_model()
    return model.encode(questions, convert_to_tensor=True)

# Fallback to OpenAI GPT-3.5-turbo for unmatched queries
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
    except Exception:
        return "Sorry, I couldn't reach the server. Please try again later."

# Find best matching answer and related questions
def find_response(user_input, dataset, embeddings, threshold=0.4):
    model = load_model()
    user_input_proc = preprocess_text(user_input)
    greetings = ["hi", "hello", "hey", "hi there", "greetings", "how are you"]
    if user_input_proc.lower() in greetings:
        return random.choice(["Hello!", "Hi there!", "Hey!", "Greetings!"]), None, 1.0, []

    user_embedding = model.encode(user_input_proc, convert_to_tensor=True)
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
st.set_page_config(page_title="🎓 Crescent University Chatbot", page_icon="🎓", layout="centered")

st.markdown("""
<style>
    .chat-message-user {
        background-color: #cce5ff;
        padding: 12px;
        border-radius: 12px;
        margin-bottom: 10px;
        font-weight: 500;
        align-self: flex-end;
        background-color: #d1eaff;
        color: #000;
    }
    .chat-message-assistant {
        background-color: #e2e3e5;
        padding: 12px;
        border-radius: 12px;
        margin-bottom: 10px;
        font-weight: 600;
        align-self: flex-start;
        background-color: #e2e2e2;
        color: #000;
    }
    st.sidebar.stbutton> Clear Chat {
        background-color: #4caf50;
        color: white;
        font-weight: bold;
    }
    body {
      background: #f5f7fa;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      margin: 0;
      padding: 0;
      display: flex;
      flex-direction: column;
      height: 100vh;
    }
    header {
      background-color: #3a7bd5;
      color: white;
      padding: 1rem;
      text-align: center;
      font-size: 1.5rem;
    }
    #chat {
      flex-grow: 1;
      padding: 1rem;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
    }
    .bubble {
      max-width: 70%;
      margin: 0.3rem 0;
      padding: 0.7rem;
      border-radius: 15px;
      line-height: 1.4;
    }
    #input-area {
      display: flex;
      padding: 1rem;
      background-color: #fff;
      border-top: 1px solid #ddd;
    }
    input {
      flex-grow: 1;
      padding: 0.6rem;
      border: 1px solid #ccc;
      border-radius: 20px;
      outline: none;
    }
    button {
      margin-left: 0.5rem;
      background-color: #3a7bd5;
      color: white;
      border: none;
      padding: 0.6rem 1rem;
      border-radius: 20px;
      cursor: pointer;
    }
    button:hover {
      background-color: #336cc9;
    }
    .typing {
      font-style: italic;
      font-size: 0.9rem;
      color: #999;
    }
  </style>
""", unsafe_allow_html=True)

st.title("🎓 Crescent University Chatbot")

model = load_model()
dataset = load_data()
question_list = dataset['question'].tolist()
question_embeddings = compute_question_embeddings(question_list)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Handle clearing chat
with st.sidebar:
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []

# Handle prefilled question (from related questions)
if "prefill_question" in st.session_state:
    prompt = st.session_state.pop("prefill_question")
else:
    prompt = st.chat_input("Ask me anything about Crescent University...")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# If user submitted a prompt
if prompt:
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Find response
    response, department, confidence, related = find_response(prompt, dataset, question_embeddings)

    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
        if department:
            st.info(f"📘 Department: **{department}**")

    # Add assistant message to history
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Limit chat history length
    if len(st.session_state.chat_history) > 50:
        st.session_state.chat_history = st.session_state.chat_history[-50:]

    # Show related questions outside chat messages for interactivity
    if related:
        selected_related = st.selectbox(
            "💡 Related questions you can ask:",
            [""] + random.sample(related, k=min(3, len(related))),
            key="related_questions"
        )
        if selected_related and selected_related != "":
            st.session_state.prefill_question = selected_related
            st.experimental_rerun()
