# ✅ Crescent University Chatbot with Styling, Paraphrase Expansion, and Topic Filtering

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

# 🎯 Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 📚 Load dataset
with open("qa_dataset.json", "r") as f:
    data = json.load(f)

# 🏷 Auto-tag by topic
for qa in data:
    q = qa["question"].lower()
    if any(w in q for w in ["fee", "payment", "tuition", "bank"]): qa["topic"] = "finance"
    elif any(w in q for w in ["hostel", "accommodation", "reside"]): qa["topic"] = "hostel"
    elif any(w in q for w in ["admission", "matriculation", "entry", "acceptance"]): qa["topic"] = "admission"
    elif any(w in q for w in ["course", "subject", "curriculum", "semester", "level"]): qa["topic"] = "academics"
    elif any(w in q for w in ["staff", "lecturer", "professor", "teacher"]): qa["topic"] = "staff"
    elif any(w in q for w in ["exam", "graduation", "malpractice"]): qa["topic"] = "exam"
    elif any(w in q for w in ["vision", "philosophy", "mission"]): qa["topic"] = "university"
    else: qa["topic"] = "general"

# 🔠 SymSpell correction
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dict_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dict_path, 0, 1)

# 🔤 Abbreviation + Synonym Maps
abbreviations = {"u": "you", "r": "are", "pls": "please", "abt": "about", "yr": "year", "sem": "semester"}
synonym_map = {"lecturers": "academic staff", "professors": "academic staff", "hod": "dean", "course": "subject"}

def normalize_text(text):
    text = text.lower()
    for abbr, full in abbreviations.items(): text = re.sub(rf'\b{re.escape(abbr)}\b', full, text)
    for key, val in synonym_map.items(): text = re.sub(rf'\b{re.escape(key)}\b', val, text)
    suggest = sym_spell.lookup_compound(text, max_edit_distance=2)
    return suggest[0].term if suggest else text

# 🧠 Load embedding model
@st.cache_resource
def load_model(): return SentenceTransformer("all-MiniLM-L6-v2")
model = load_model()

# 🔍 Build FAISS index
@st.cache_resource
def build_index():
    questions = [normalize_text(qa["question"]) for qa in data]
    emb = model.encode(questions, show_progress_bar=False)
    emb = np.array(emb).astype("float32")
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(emb.shape[1]), emb.shape[1], 100)
    index.train(emb)
    index.add(emb)
    return index, emb, questions

index, embeddings, questions = build_index()

# 🤖 GPT fallback

def rag_fallback_with_context(query, top_k_matches):
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
        context_parts, total_tokens = [], 0
        for i in top_k_matches:
            if i < len(data):
                pair = f"Q: {data[i]['question']}\nA: {data[i]['answer']}"
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

# 💬 Message UI styling

def render_message(msg, is_user):
    return f"""
    <div style='background-color:{'#DCF8C6' if is_user else '#E1E1E1'}; padding:10px; 
         margin:{'10px 0 10px 50px' if is_user else '10px 50px 10px 0'}; 
         border-radius:10px; max-width:70%; font-family:sans-serif; font-size:15px;'>
        {msg}
    </div>"""

# 🧾 Streamlit App
st.set_page_config(page_title="Crescent Chatbot", layout="centered")
st.title("🎓 Crescent University Chatbot")
st.markdown("Ask me anything about admissions, courses, hostels, or more!")

if "history" not in st.session_state:
    st.session_state.history = []

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", key="user_input", placeholder="Type your question here...")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    norm_input = normalize_text(user_input)
    st.session_state.history.append((user_input, True))

    query_vec = model.encode([norm_input]).astype("float32")
    index.nprobe = 10
    D, I = index.search(query_vec, k=3)
    score, idx = D[0][0], I[0][0]

    if score > 1.0 or np.isnan(score):
        response = rag_fallback_with_context(norm_input, I[0])
    else:
        response = data[idx]["answer"]

    if not response.endswith(('.', '!', '?')): response += '.'
    response = "Sure! " + response[0].upper() + response[1:]
    st.session_state.history.append((response, False))

# 🖼 Display messages
for msg, is_user in st.session_state.history:
    st.markdown(render_message(msg, is_user), unsafe_allow_html=True)
