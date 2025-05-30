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

# Set page config FIRST thing
st.set_page_config(page_title="Crescent Chatbot", layout="centered")

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Load structured dataset
with open("qa_dataset.json", "r") as f:
    data = json.load(f)

# Topic filter UI
topics = sorted(set(q["topic"] for q in data))
selected_topic = st.selectbox("Filter by topic", ["All"] + topics)

# Filter dataset by topic
if selected_topic != "All":
    filtered_data = [q for q in data if q["topic"] == selected_topic]
else:
    filtered_data = data

# Initialize SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Abbreviation mapping (add all you want here)
abbreviations = {
    "u": "you", "r": "are", "ur": "your", "ow": "how", "pls": "please", "plz": "please",
    "tmrw": "tomorrow", "cn": "can", "wat": "what", "cud": "could", "shud": "should",
    "wud": "would", "abt": "about", "bcz": "because", "btw": "between", "asap": "as soon as possible",
    "idk": "i don't know", "imo": "in my opinion", "msg": "message", "doc": "document", "d": "the",
    "yr": "year", "sem": "semester", "dept": "department", "admsn": "admission",
    "cresnt": "crescent", "uni": "university", "clg": "college", "sch": "school",
    "info": "information", "l": "level", "CSC": "Computer Science", "ECO": "Economics with Operations Research",
    "PHY": "Physics", "STAT": "Statistics", "1st": "first", "2nd": "second", 
    "tech staff": "technical staff", "it people": "technical staff", "lab helper": "technical staff",
    "computer staff": "technical staff", "equipment handler": "technical staff",
    "office staff": "non-academic staff", "admin worker": "non-academic staff",
    "support staff": "non-academic staff", "clerk": "non-academic staff", "receptionist": "non-academic staff",
    "school worker": "non-academic staff", "it guy": "technical staff", "secretary": "non-academic staff"
}

synonym_map = {
    "lecturers": "academic staff", "professors": "academic staff", "teachers": "academic staff", "instructors": "academic staff",
    "tutors": "academic staff", "head": "dean", "school": "university", "course": "subject", "class": "course",
    "tech staff": "technical staff", "it people": "technical staff", "lab helper": "technical staff", "computer staff": "technical staff",
    "equipment handler": "technical staff", "office staff": "non-academic staff", "admin worker": "non-academic staff",
    "support staff": "non-academic staff", "clerk": "non-academic staff", "receptionist": "non-academic staff",
    "school worker": "non-academic staff", "it guy": "technical staff", "secretary": "non-academic staff"
}

def normalize_text(text):
    text = text.lower()
    for abbr, full in abbreviations.items():
        text = re.sub(rf'\b{re.escape(abbr)}\b', full, text)
    for key, val in synonym_map.items():
        text = re.sub(rf'\b{re.escape(key)}\b', val, text)
    suggest = sym_spell.lookup_compound(text, max_edit_distance=2)
    return suggest[0].term if suggest else text

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

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

def rag_fallback_with_context(query, top_k_matches):
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
        context_parts, total_tokens = [], 0
        for i in top_k_matches:
            if i < len(filtered_data):
                pair = f"Q: {filtered_data[i]['question']}\nA: {filtered_data[i]['answer']}"
                tokens = len(encoding.encode(pair))
                if total_tokens + tokens > 3596:
                    break
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

def search_course_departments(course_query):
    course_query = course_query.lower()
    matches = [q for q in data if course_query in q["question"].lower() and q.get("level") == "100"]
    if matches:
        depts = sorted(set(m["department"] for m in matches if m.get("department")))
        facs = sorted(set(m["faculty"] for m in matches if m.get("faculty")))
        return f"This course is taken in departments: {', '.join(depts)} under faculties: {', '.join(facs)}."
    return "Course not found across departments."

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

        if not response.endswith(('.', '!', '?')):
            response += '.'
        response = "Sure! " + response[0].upper() + response[1:]
        st.session_state.history.append((response, False))

# Display conversation
for msg, is_user in st.session_state.history:
    st.markdown(render_message(msg, is_user), unsafe_allow_html=True)
