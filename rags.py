import json
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from symspellpy.symspellpy import SymSpell
import numpy as np
import os
import re
import random
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client = OpenAI()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# Load SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", 0, 1)

# Load dataset
with open("qa_dataset.json") as f:
    data = json.load(f)

# Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[\W_]+", " ", text)
    return text.strip()

# Apply preprocessing to dataset questions
questions = [preprocess_text(item['question']) for item in data]
question_embeddings = model.encode(questions, convert_to_tensor=True)

# Spell correction
def correct_spelling(text):
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    return suggestions[0].term if suggestions else text

# Semantic search with metadata
def find_response(user_input, dataset, embeddings, threshold=0.65):
    cleaned = correct_spelling(preprocess_text(user_input))
    user_embedding = model.encode(cleaned, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(user_embedding, embeddings)[0]
    top_idx = int(np.argmax(similarities))
    top_score = float(similarities[top_idx])
    if top_score >= threshold:
        item = dataset[top_idx]
        answer = item['answer']
        dept = item.get('department', '')
        level = item.get('level', '')
        extra = ""
        if dept or level:
            extra += f"\n\n**Department**: {dept if dept else 'N/A'}"
            extra += f"\n**Level**: {level if level else 'N/A'}"
        return answer + extra
    return None

# GPT fallback (OpenAI v1+)
def gpt_fallback(user_input):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful chatbot for Crescent University."},
            {"role": "user", "content": user_input}
        ]
    )
    return response.choices[0].message.content.strip()

# Greeting logic
greeting_inputs = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "greetings"]
greeting_responses = [
    "Hello! How can I help you today?",
    "Hi there! Ask me anything about Crescent University.",
    "Hey! What would you like to know?",
    "Greetings! I'm here to assist you.",
    "Hi! How can I be of service?"
]

def is_greeting(text):
    cleaned = text.lower().strip()
    return any(greet in cleaned for greet in greeting_inputs)

# UI message styling
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

# Streamlit app
st.title("Crescent University Chatbot")
st.markdown("Ask me anything about Crescent University 🧠")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Your question:", key="input")

if user_input:
    with st.spinner("Thinking..."):
        if is_greeting(user_input):
            response = random.choice(greeting_responses)
        else:
            response = find_response(user_input, data, question_embeddings)
            if not response:
                response = gpt_fallback(user_input)
        st.session_state.history.append({
            "user": user_input,
            "bot": response
        })

# Display chat history
for chat in st.session_state.history:
    st.markdown(render_message(chat["user"], is_user=True), unsafe_allow_html=True)
    st.markdown(render_message(chat["bot"], is_user=False), unsafe_allow_html=True)
