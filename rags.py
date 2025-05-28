import json
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from symspellpy.symspellpy import SymSpell
import numpy as np
import openai
import os
from dotenv import load_dotenv
import re

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

model = SentenceTransformer('all-MiniLM-L6-v2')

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", 0, 1)

with open("qa_dataset.json") as f:
    data = json.load(f)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[\W_]+", " ", text)
    return text.strip()

def correct_spelling(text):
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    return suggestions[0].term if suggestions else text

questions = [item['question'] for item in data]
question_embeddings = model.encode(questions, convert_to_tensor=True)

def find_response(user_input, dataset, embeddings, threshold=0.65):
    cleaned = correct_spelling(preprocess_text(user_input))
    user_embedding = model.encode(cleaned, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(user_embedding, embeddings)[0]
    top_idx = int(np.argmax(similarities))
    top_score = float(similarities[top_idx])
    if top_score >= threshold:
        return dataset[top_idx]['answer']
    else:
        return None

def gpt_fallback(user_input):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful chatbot for Crescent University."},
            {"role": "user", "content": user_input}
        ]
    )
    return response.choices[0].message['content'].strip()

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
        color: #000;
    ">
        {message}
    </div>
    """

st.title("Crescent University Chatbot")
st.markdown("Ask me anything about Crescent University 🧠")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Your question:", key="input")

if user_input:
    with st.spinner("Thinking..."):
        response = find_response(user_input, data, question_embeddings)
        if not response:
            response = gpt_fallback(user_input)
        st.session_state.history.append({
            "user": user_input,
            "bot": response
        })

for chat in st.session_state.history:
    st.markdown(render_message(chat["user"], is_user=True), unsafe_allow_html=True)
    st.markdown(render_message(chat["bot"], is_user=False), unsafe_allow_html=True)

st.markdown("---")
st.markdown("**Examples:**")
st.markdown("- What is the vision of Crescent University?")
st.markdown("- Who is the Vice Chancellor?")
st.markdown("- What is the tuition fee for new students?")
