# Crescent University Chatbot (Clean Version)

# --- 1. Imports & Initial Setup ---
import streamlit as st
import pandas as pd
import torch
import random
import re
import json
import os
import uuid
import hashlib

from sentence_transformers import SentenceTransformer, util
from symspellpy.symspellpy import SymSpell, Verbosity
import pkg_resources
import openai

# --- 2. API Key & Spellcheck Setup ---
openai.api_key = os.getenv("OPENAI_API_KEY")

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# --- 3. Abbreviation & Department Maps ---
abbreviations = {
    "uni": "university", "dept": "department", "cuab": "crescent university", "lec": "lecture",
    "sch": "school", "asgn": "assignment", "proj": "project", "exm": "exam", "smt": "semester",
    "lvl": "level", "yrs": "years", "yrs": "years", "sem": "semester"
}

department_map = {
    "PHY": "Physics", "BIO": "Biology", "CHM": "Chemistry", "CSC": "Computer Science",
    "MAT": "Mathematics", "STA": "Statistics", "ECO": "Economics", "ACC": "Accounting",
    "BUS": "Business Administration", "MKT": "Marketing", "PSY": "Psychology", "SOC": "Sociology",
    "POL": "Political Science", "ENG": "English", "HIS": "History", "IR": "International Relations"
}

# --- 4. Text Processing Functions ---
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

# --- 5. Load Model & Data ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    with open("qa_dataset.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    return pd.DataFrame([
        {
            "text": f"Q: {q}\nA: {a}",
            "question": q,
            "answer": a,
            "department": d,
            "level": l,
            "semester": s,
            "faculty": f
        }
        for entry in raw_data
        if (q := entry.get("question", "").strip()) and (a := entry.get("answer", "").strip())
        and (d := entry.get("department", "").strip()) is not None
        and (l := entry.get("level", "").strip()) is not None
        and (s := entry.get("semester", "").strip()) is not None
        and (f := entry.get("faculty", "").strip()) is not None
    ])

@st.cache_data
def compute_question_embeddings(questions):
    model = load_model()
    return model.encode(questions, convert_to_tensor=True)

# --- 6. Fallback OpenAI GPT ---
def fallback_openai(user_input, context_qa=None):
    system_prompt = (
        "You are a helpful assistant specialized in Crescent University information. "
        "If you don't know an answer, politely say so and refer to university resources."
    )
    messages = [{"role": "system", "content": system_prompt}]
    user_message = (f"Here is some relevant university information:\nQ: {context_qa['question']}\nA: {context_qa['answer']}\n\n"
                    if context_qa else "") + f"Answer this question: {user_input}"
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

# --- 7. Search & Retrieval ---
def find_response(user_input, dataset, embeddings, threshold=0.4):
    model = load_model()
    user_input_clean = preprocess_text(user_input)

    greetings = ["hello", "hi", "hey", "good day", "howdy"]
    if user_input_clean.lower() in greetings:
        return random.choice(["Hello!", "Hi there!", "Welcome to Crescent University chatbot!", "Hey, how can I help you?"]), None, 1.0, []

    user_embedding = model.encode(user_input_clean, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(user_embedding, embeddings)[0]
    top_scores, top_indices = torch.topk(cos_scores, k=5)

    top_score = top_scores[0].item()
    top_index = top_indices[0].item()

    if top_score < threshold:
        context_qa = dataset.iloc[top_index][["question", "answer"]].to_dict()
        return fallback_openai(user_input, context_qa), None, top_score, []

    row = dataset.iloc[top_index]
    related_questions = [dataset.iloc[i.item()]["question"] for i in top_indices[1:]]

    match = re.search(r"\b([A-Z]{2,}-?\d{3,})\b", row["question"])
    department = department_map.get(extract_prefix(match.group(1)), "Unknown") if match else None

    if random.random() < 0.2:
        row["answer"] = random.choice(["I think ", "Maybe: ", "Possibly: ", "Here's what I found: "]) + row["answer"]

    return row["answer"], department, top_score, related_questions

# --- 8. Filter Logic ---
def apply_filters(df, faculty, department, level, semester):
    if faculty: df = df[df['faculty'].isin(faculty)]
    if department: df = df[df['department'].isin(department)]
    if level: df = df[df['level'].isin(level)]
    if semester: df = df[df['semester'].isin(semester)]
    return df

# --- 9. UI Styling ---
st.set_page_config(page_title="Crescent University Chatbot", page_icon="🎓")
st.markdown("""
<style>
.chat-message-user {
    background-color: #d6eaff;
    padding: 12px;
    border-radius: 15px;
    margin-bottom: 10px;
    margin-left: auto;
    max-width: 75%;
    font-weight: 550;
    color: #000;
    word-wrap: break-word;
}
.chat-message-assistant {
    background-color: #f5f5f5;
    margin-right: auto;
    margin-bottom: 10px;
    font-weight: 600;
    color: #000;
    text-align: left;
    word-wrap: break-word;
}
.department-label {
    font-size: 0.85em;
    color: #444; /* Improved contrast */
    margin-top: -10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🎓 Crescent University Chatbot")

# --- 10. State Initialization ---
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "related_questions" not in st.session_state: st.session_state.related_questions = []
if "last_department" not in st.session_state: st.session_state.last_department = None

# --- 11. Load Data & Filter UI ---
model = load_model()
dataset = load_data()

with st.sidebar:
    st.header("Filter Questions")
    selected_faculty = st.multiselect("Faculty", sorted(dataset['faculty'].dropna().unique()))
    selected_department = st.multiselect("Department", sorted(dataset['department'].dropna().unique()))
    selected_level = st.multiselect("Level", sorted(dataset['level'].dropna().unique()))
    selected_semester = st.multiselect("Semester", sorted(dataset['semester'].dropna().unique()))

filtered_dataset = apply_filters(dataset, selected_faculty, selected_department, selected_level, selected_semester)
question_list = filtered_dataset['question'].tolist()
question_embeddings = compute_question_embeddings(question_list) if not filtered_dataset.empty else None

with st.sidebar:
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.related_questions = []
        st.session_state.last_department = None
        st.rerun()

# --- 12. Display Chat History ---
for message in st.session_state.chat_history:
    role_class = "chat-message-user" if message["role"] == "user" else "chat-message-assistant"
    with st.chat_message(message["role"]):
        st.markdown(f'<div class="{role_class}">{message["content"]}</div>', unsafe_allow_html=True)
        if message["role"] == "assistant" and st.session_state.last_department:
            st.markdown(f'<div class="department-label">Department: {st.session_state.last_department}</div>', unsafe_allow_html=True)

# --- 13. Chat Input ---
prompt = st.chat_input("Ask me anything about Crescent University...")
if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    matched_row = dataset[dataset['question'].str.lower() == prompt.lower()]
    if not matched_row.empty:
        answer = matched_row.iloc[0]['answer']
        department, related = None, []
    else:
        answer, department, score, related = find_response(prompt, filtered_dataset, question_embeddings)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.session_state.related_questions = related
    st.session_state.last_department = department
    st.rerun()

# --- 14. Related Suggestions ---
if st.session_state.related_questions:
    st.markdown("#### 💡 You might also ask:")
    for q in st.session_state.related_questions:
        unique_key = f"{uuid.uuid4().hex}"
        if st.button(q, key=f"related_{unique_key}", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": q})
            answer, department, score, related = find_response(q, dataset, question_embeddings)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.session_state.related_questions = related
            st.session_state.last_department = department
            st.rerun()
