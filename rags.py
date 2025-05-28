# Improved version of rags.py with semantic similarity fixes and better robustness

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

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load Sentence Transformer Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", 0, 1)

# Load Q&A dataset
with open("qa_dataset.json") as f:
    data = json.load(f)

# Preprocess

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[\W_]+", " ", text)
    return text.strip()

# Spell correction

def correct_spelling(text):
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    return suggestions[0].term if suggestions else text

# Embed dataset questions
questions = [item['question'] for item in data]
question_embeddings = model.encode(questions, convert_to_tensor=True)

# Semantic search

def find_response(user_input, dataset, embeddings, threshold=0.65):
    # Clean and correct input
    cleaned = correct_spelling(preprocess_text(user_input))
    user_embedding = model.encode(cleaned, convert_to_tensor=True)

    # Compute similarity
    similarities = util.pytorch_cos_sim(user_embedding, embeddings)[0]
    top_idx = int(np.argmax(similarities))
    top_score = float(similarities[top_idx])

    if top_score >= threshold:
        return dataset[top_idx]['answer']
    else:
        return None

# GPT fallback

def gpt_fallback(user_input):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful chatbot for Crescent University."},
            {"role": "user", "content": user_input}
        ]
    )
    return response.choices[0].message['content'].strip()

# Streamlit UI
st.title("Crescent University Chatbot")
st.markdown("Ask me anything about Crescent University 🧠")

user_input = st.text_input("Your question:", key="input")
if user_input:
    with st.spinner("Thinking..."):
        response = find_response(user_input, data, question_embeddings)
        if not response:
            response = gpt_fallback(user_input)
        st.success(response)
