# --- main.py ---
import streamlit as st
import uuid
from utils.preprocessing import preprocess_text, normalize_text
from utils.rag_engine import (
    load_model, load_data, compute_question_embeddings,
    find_response, fallback_openai
)
from utils.constants import abbreviations, department_map

st.set_page_config(page_title="Crescent University Chatbot", page_icon="🎓")

model = load_model()
dataset = load_data()
question_list = dataset['question'].tolist()
question_embeddings = compute_question_embeddings(question_list)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "related_questions" not in st.session_state:
    st.session_state.related_questions = []
if "last_department" not in st.session_state:
    st.session_state.last_department = None

# --- Sidebar ---
with st.sidebar:
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.related_questions = []
        st.session_state.last_department = None
        st.rerun()

# --- Title and Styles ---
st.markdown("""
<style>
    html, body, .stApp { font-family: 'Open Sans', sans-serif; }
    h1, h2, h3, h4, h5 { font-family: 'Merriweather', serif; color: #004080; }
    .chat-message-user {
        background-color: #d6eaff;
        padding: 12px;
        border-radius: 15px;
        margin-bottom: 10px;
        margin-left: auto;
        max-width: 75%;
        font-weight: 550;
        color: #000;
    }
    .chat-message-assistant {
        background-color: #f5f5f5;
        padding: 12px;
        border-radius: 15px;
        margin-bottom: 10px;
        margin-right: auto;
        max-width: 75%;
        font-weight: 600;
        color: #000;
    }
    .related-question {
        background-color: #e6f2ff;
        padding: 8px 12px;
        margin: 6px 6px 6px 0;
        display: inline-block;
        border-radius: 10px;
        font-size: 0.9rem;
        cursor: pointer;
    }
    .department-label {
        font-family: 'Merriweather', serif;
        font-size: 0.85rem;
        color: #004080;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

st.title(":mortar_board: Crescent University Chatbot")

# --- Chat History Render ---
for message in st.session_state.chat_history:
    role_class = "chat-message-user" if message["role"] == "user" else "chat-message-assistant"
    with st.chat_message(message["role"]):
        st.markdown(f'<div class="{role_class}">{message["content"]}</div>', unsafe_allow_html=True)
        if message["role"] == "assistant" and st.session_state.last_department:
            dept_label = st.session_state.last_department.title()
            st.markdown(f'<div class="department-label">Department: {dept_label}</div>', unsafe_allow_html=True)

# --- Chat Input ---
prompt = st.chat_input("Ask me anything about Crescent University...")

if prompt and prompt.strip():
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    matched_row = dataset[dataset['question'].str.lower() == prompt.lower()]
    if not matched_row.empty:
        answer = matched_row.iloc[0]['answer']
        department = None
        related = []
    else:
        answer, department, score, related = find_response(prompt, dataset, question_embeddings)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.session_state.related_questions = related
    st.session_state.last_department = department
    st.rerun()

# --- Related Questions ---
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
