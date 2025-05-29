import json
import streamlit as st
from sentence_transformers import SentenceTransformer
from symspellpy.symspellpy import SymSpell, Verbosity
import numpy as np
import os
import re
import random
import pkg_resources
from dotenv import load_dotenv
import openai
import faiss
from collections import defaultdict

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("OpenAI API key not found in environment variables")

model = SentenceTransformer('all-mpnet-base-v2', device='cpu')

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

abbreviations = {
    "u": "you", "r": "are", "ur": "your", "ow": "how", "pls": "please", "plz": "please",
    "tmrw": "tomorrow", "cn": "can", "wat": "what", "cud": "could", "shud": "should",
    "wud": "would", "abt": "about", "bcz": "because", "btw": "between", "asap": "as soon as possible",
    "idk": "i don't know", "imo": "in my opinion", "msg": "message", "doc": "document", "d": "the",
    "yr": "year", "sem": "semester", "dept": "department", "admsn": "admission",
    "cresnt": "crescent", "uni": "university", "clg": "college", "sch": "school",
    "info": "information", "l": "level"
}

synonym_map = {
    "lecturers": "academic staff", "professors": "academic staff", "teachers": "academic staff", "instructors": "academic staff",
    "tutors": "academic staff", "head": "dean", "school": "university", "course": "subject", "class": "course"
}

def normalize_text(text):
    text = re.sub(r'([^a-zA-Z0-9\s])', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r"\bwho’s\b", "who is", text)
    text = re.sub(r"\bwhat’s\b", "what is", text)
    text = re.sub(r"\bhow’s\b", "how is", text)
    return text

def preprocess_text(text):
    text = normalize_text(text.lower())
    for phrase, replacement in {**abbreviations, **synonym_map}.items():
        if phrase in text:
            text = text.replace(phrase, replacement)
    words = text.split()
    expanded = []
    for word in words:
        word = abbreviations.get(word, word)
        word = synonym_map.get(word, word)
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        corrected = suggestions[0].term if suggestions else word
        expanded.append(corrected)
    return ' '.join(expanded)

with open("qa_dataset.json") as f:
    data = json.load(f)

data_by_dept = defaultdict(list)
indices_by_dept = {}

for item in data:
    dept = item.get("department", "All")
    data_by_dept[dept].append(item)

for dept, dept_data in data_by_dept.items():
    questions = [preprocess_text(q["question"]) for q in dept_data]
    embeddings = model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    indices_by_dept[dept] = (index, dept_data)

def find_top_k_matches(user_input, dataset, index, top_k=3):
    cleaned = preprocess_text(user_input)
    user_embedding = model.encode([cleaned], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(user_embedding, top_k)
    top_k_matches = []
    for i, idx in enumerate(indices[0]):
        top_k_matches.append({
            "question": dataset[idx]['question'],
            "answer": dataset[idx]['answer'],
            "score": float(scores[0][i])
        })
    return top_k_matches

def gpt_fallback_with_context(user_input, top_matches):
    context_blocks = "\n".join([
        f"{i+1}. {item['question']} — {item['answer']}" for i, item in enumerate(top_matches)
    ])
    messages = [
        {"role": "system", "content": "You're a helpful assistant for Crescent University. Answer based only on the given context."},
        {"role": "user", "content": f"""Context:
{context_blocks}

User Question: {user_input}
Answer based strictly on the context above."""}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=messages,
        timeout=10
    )
    return response['choices'][0]['message']['content']

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

st.title("🎓 Crescent University Chatbot")
st.markdown("Ask me anything about Crescent University!")

if "history" not in st.session_state:
    st.session_state.history = []

selected_dept = st.selectbox("Filter by department (optional):", options=["All"] + list(indices_by_dept.keys()))
user_input = st.text_input("Your question:", key="input")

if user_input:
    with st.spinner("Thinking..."):
        dept = selected_dept if selected_dept in indices_by_dept else "All"
        index, filtered_data = indices_by_dept[dept]

        top_matches = find_top_k_matches(user_input, filtered_data, index, top_k=3)
        best_match = top_matches[0]

        if best_match['score'] >= 0.6:
            final_response = best_match['answer']
        else:
            try:
                final_response = gpt_fallback_with_context(user_input, top_matches)
            except Exception:
                final_response = "I'm not sure how to answer that. Please try rephrasing."

        st.session_state.history.append({"user": user_input, "bot": final_response})

for chat in st.session_state.history:
    st.markdown(render_message(chat["user"], is_user=True), unsafe_allow_html=True)
    st.markdown(render_message(chat["bot"], is_user=False), unsafe_allow_html=True)

# Optional: Batch Q&A
st.markdown("---")
st.markdown("### 🧪 Batch Q&A")
multi_input = st.text_area("Enter multiple questions (one per line):")
if st.button("Submit Batch"):
    questions = [q.strip() for q in multi_input.strip().split("\n") if q.strip()]
    for q in questions:
        dept = selected_dept if selected_dept in indices_by_dept else "All"
        index, filtered_data = indices_by_dept[dept]
        top_matches = find_top_k_matches(q, filtered_data, index, top_k=3)
        if top_matches[0]['score'] >= 0.6:
            response = top_matches[0]['answer']
        else:
            try:
                response = gpt_fallback_with_context(q, top_matches)
            except Exception:
                response = "I'm not sure how to answer that."
        st.markdown(render_message(q, is_user=True), unsafe_allow_html=True)
        st.markdown(render_message(response, is_user=False), unsafe_allow_html=True)
