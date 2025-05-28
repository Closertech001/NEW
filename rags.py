import json
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from symspellpy.symspellpy import SymSpell
import numpy as np
import os
import re
import random
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Get API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please set it in your .env file.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# Load sentence transformer model on CPU
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# Initialize SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", 0, 1)

# Load dataset
with open("qa_dataset.json") as f:
    data = json.load(f)

questions = [item['question'] for item in data]
question_embeddings = model.encode(questions, convert_to_tensor=True)

# Helpers
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[\W_]+", " ", text)
    return text.strip()

def correct_spelling(text):
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    return suggestions[0].term if suggestions else text

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
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful chatbot for Crescent University."},
            {"role": "user", "content": user_input}
        ]
    )
    return response.choices[0].message.content.strip()

# Greetings
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
    return cleaned in greeting_inputs

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

# Streamlit UI
st.title("🎓 Crescent University Chatbot")
st.markdown("Ask me anything about Crescent University!")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Your question:", key="input")

if user_input:
    with st.spinner("Thinking..."):
        if is_greeting(user_input):
            response = random.choice(greeting_responses)
        else:
            response = find_response(user_input, data, question_embeddings)
            if response:
                final_response = response
            else:
                try:
                    final_response = gpt_fallback(user_input)
                except Exception:
                    final_response = (
                        "I'm not sure how to answer that at the moment. "
                        "Could you try asking in a different way or with more details?"
                    )

        st.session_state.history.append({
            "user": user_input,
            "bot": response
        })

# Display chat history
for chat in st.session_state.history:
    st.markdown(render_message(chat["user"], is_user=True), unsafe_allow_html=True)
    st.markdown(render_message(chat["bot"], is_user=False), unsafe_allow_html=True)
