import streamlit as st

from utils.rag_engine import (
    load_data,
    compute_question_embeddings,
    find_response,
)

st.set_page_config(
    page_title="Crescent University Chatbot",
    page_icon="🎓",
)

# --- Load cached university data and embeddings ---
dataset = load_data()
question_list = dataset["question"].tolist()
question_embeddings = compute_question_embeddings(question_list)

# --- Initialize chat history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar ---
with st.sidebar:
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# --- Styling ---
st.markdown(
    """
    <style>
        html, body, .stApp {
            font-family: 'Open Sans', sans-serif;
        }
        h1, h2, h3, h4, h5 {
            font-family: 'Merriweather', serif;
            color: #004080;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎓 Crescent University Chatbot")

# --- Render chat history ---
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat input ---
prompt = st.chat_input(
    "Ask me anything about Crescent University..."
)

if prompt and prompt.strip():
    st.session_state.chat_history.append(
        {"role": "user", "content": prompt}
    )

    matched_row = dataset[
        dataset["question"].str.strip().str.casefold()
        == prompt.strip().casefold()
    ]

    if not matched_row.empty:
        answer = matched_row.iloc[0]["answer"]
    else:
        answer, department, score, related = find_response(
            prompt,
            dataset,
            question_embeddings,
        )

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer}
    )
    st.rerun()