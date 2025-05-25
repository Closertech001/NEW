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

# Set your OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load SymSpell for spell correction
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Abbreviations mapping for normalization
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

# Normalize text by removing special chars and repeated letters
def normalize_text(text):
    text = re.sub(r'([^a-zA-Z0-9\s])', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    return text

# Preprocess input: normalize, expand abbreviations, autocorrect
def preprocess_text(text):
    text = normalize_text(text)
    words = text.split()
    expanded = [abbreviations.get(word.lower(), word) for word in words]
    corrected = []
    for word in expanded:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        corrected.append(suggestions[0].term if suggestions else word)
    return ' '.join(corrected)

# Extract department prefix from course code
def extract_prefix(code):
    match = re.match(r"([A-Z\-]+)", code)
    return match.group(1) if match else None

# Load embedding model (cached for performance)
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load your Q&A dataset (assumed to be a list of dicts with keys: question, answer)
@st.cache_resource
def load_data():
    with open("qa_dataset.json", "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
    return pd.DataFrame(qa_pairs)

# Compute embeddings for questions once (cached)
@st.cache_data
def compute_question_embeddings(questions: list):
    model = load_model()
    return model.encode(questions, convert_to_tensor=True)

# GPT fallback with context from a Q&A pair for better responses
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

# Find best response given user input and dataset
def find_response(user_input, dataset, embeddings, threshold=0.4):
    model = load_model()
    user_input_clean = preprocess_text(user_input)

    greetings = ["hi", "hello", "hey", "hi there", "greetings", "how are you"]
    if user_input_clean.lower() in greetings:
        return random.choice(["Hello!", "Hi there!", "Hey!", "Greetings!"]), None, 1.0, []

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
        uncertainty = random.choice(["I think ", "Maybe: ", "Possibly: ", "Here's what I found: "])
        response = uncertainty + response

    return response, department, top_score, related_questions

# --- Streamlit UI setup ---
st.set_page_config(page_title="🎓 Crescent University Chatbot", page_icon="🎓")
st.markdown("""
<style>
    .chat-message-user {
        background-color: #cce5ff;
        padding: 12px;
        border-radius: 12px;
        margin-bottom: 10px;
        font-weight: 550;
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
        align-self: flex-end;
        background-color: #d1eaff;
        color: #000;
    }
    .sidebar .stButton>button {
        background-color: #4caf50;
        color: white;
        font-weight: bold;
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

if "prefill_question" in st.session_state:
    prompt = st.session_state.pop("prefill_question")
else:
    prompt = st.chat_input("Ask me anything about Crescent University...")

with st.sidebar:
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    role_class = "chat-message-user" if message["role"] == "user" else "chat-message-assistant"
    with st.chat_message(message["role"]):
        st.markdown(f'<div class="{role_class}">{message["content"]}</div>', unsafe_allow_html=True)

if prompt is not None:
    with st.chat_message("user"):
        st.markdown(f'<div class="chat-message-user">{prompt}</div>', unsafe_allow_html=True)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    response, department, confidence, related = find_response(prompt, dataset, question_embeddings)

    response_md = response
    if department:
        response_md += f"\n\n<em>📘 Department: <strong>{department}</strong></em>"

    with st.chat_message("assistant"):
        st.markdown(f'<div class="chat-message-assistant">{response_md}</div>', unsafe_allow_html=True)

        if related:
            selected_related = st.selectbox(
                "💡 Related questions you can ask:",
                [""] + random.sample(related, k=min(3, len(related))),
                key=f"related_{len(st.session_state.chat_history)}"
            )
            if selected_related:
                st.session_state.prefill_question = selected_related
                st.experimental_rerun()

    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Limit chat history size
    if len(st.session_state.chat_history) > 50:
        st.session_state.chat_history = st.session_state.chat_history[-50:]
