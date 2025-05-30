# ✅ Crescent University Chatbot with Structured Dataset & Topic Filtering

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

# 🔐 Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 📚 Load structured dataset
with open("qa_dataset_cleaned_structured.json", "r") as f:
    data = json.load(f)

# 🏷 Topic filter UI
topics = sorted(set(q["topic"] for q in data))
selected_topic = st.selectbox("Filter by topic", ["All"] + topics)

# Filter dataset
if selected_topic != "All":
    filtered_data = [q for q in data if q["topic"] == selected_topic]
else:
    filtered_data = data

# 🔠 SymSpell correction
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dict_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dict_path, 0, 1)

abbreviations = {"u": "you", "r": "are", "pls": "please", "abt": "about", "yr": "year", "sem": "semester"}
synonym_map = {"lecturers": "academic staff", "professors": "academic staff", "hod": "dean", "course": "subject"}

def normalize_text(text):
    text = text.lower()
    for abbr, full in abbreviations.items(): text = re.sub(rf'\b{re.escape(abbr)}\b', full, text)
    for key, val in synonym_map.items(): text = re.sub(rf'\b{re.escape(key)}\b', val, text)
    suggest = sym_spell.lookup_compound(text, max_edit_distance=2)
    return suggest[0].term if suggest else text

@st.cache_resource
def load_model(): return SentenceTransformer("all-MiniLM-L6-v2")
model = load_model()

@st.cache_resource
def build_index():
    questions = [normalize_text(qa["question"]) for qa in filtered_data]
    emb = model.encode(questions, show_progress_bar=False)
    emb = np.array(emb).astype("float32")
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(emb.shape[1]), emb.shape[1], 100)
    index.train(emb)
    index.add(emb)
    return index, emb, questions

index, embeddings, questions = build_index()

# 📘 RAG fallback

def rag_fallback_with_context(query, top_k_matches):
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
        context_parts, total_tokens = [], 0
        for i in top_k_matches:
            if i < len(filtered_data):
                pair = f"Q: {filtered_data[i]['question']}\nA: {filtered_data[i]['answer']}"
                tokens = len(encoding.encode(pair))
                if total_tokens + tokens > 3596: break
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

# 🔍 Search shared course across departments

def search_course_departments(course_query):
    course_query = course_query.lower()
    matches = [q for q in data if course_query in q["question"].lower() and q.get("level") == "100"]
    if matches:
        depts = sorted(set(m["department"] for m in matches if m["department"]))
        facs = sorted(set(m["faculty"] for m in matches if m["faculty"]))
        return f"This course is taken in departments: {', '.join(depts)} under faculties: {', '.join(facs)}."
    return "Course not found across departments."

# 💬 Message styling

def render_message(msg, is_user):
    return f"""
    <div style='background-color:{'#DCF8C6' if is_user else '#E1E1E1'}; padding:10px;
         margin:{'10px 0 10px 50px' if is_user else '10px 50px 10px 0'};
         border-radius:10px; max-width:70%; font-family:sans-serif; font-size:15px;'>
        {msg}
    </div>"""

# 🧾 Streamlit UI
st.set_page_config(page_title="Crescent Chatbot", layout="centered")
st.title("🎓 Crescent University Chatbot")
st.markdown("Ask about admissions, courses, hostels, fees, staff, etc.")

if "history" not in st.session_state:
    st.session_state.history = []

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", key="user_input", placeholder="Type your question here...")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    norm_input = normalize_text(user_input)
    st.session_state.history.append((user_input, True))

    if "which department offers" in norm_input or "who takes" in norm_input:
        course_info = search_course_departments(norm_input)
        st.session_state.history.append((course_info, False))
    else:
        query_vec = model.encode([norm_input]).astype("float32")
        index.nprobe = 10
        D, I = index.search(query_vec, k=3)
        score, idx = D[0][0], I[0][0]

        if score > 1.0 or np.isnan(score):
            response = rag_fallback_with_context(norm_input, I[0])
        else:
            response = filtered_data[idx]["answer"]

        if not response.endswith(('.', '!', '?')): response += '.'
        response = "Sure! " + response[0].upper() + response[1:]
        st.session_state.history.append((response, False))

# 🖼 Display conversation
for msg, is_user in st.session_state.history:
    st.markdown(render_message(msg, is_user), unsafe_allow_html=True)
