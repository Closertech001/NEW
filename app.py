import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import random
import re
from symspellpy.symspellpy import SymSpell, Verbosity
import pkg_resources
import os
import json

# Load SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Abbreviation dictionary
abbreviations = {
    "u": "you", "r": "are", "ur": "your", "ow": "how", "pls": "please", "plz": "please",
    "tmrw": "tomorrow", "cn": "can", "wat": "what", "cud": "could", "shud": "should",
    "wud": "would", "abt": "about", "bcz": "because", "bcoz": "because", "btw": "between",
    "asap": "as soon as possible", "idk": "i don't know", "imo": "in my opinion",
    "msg": "message", "doc": "document", "d": "the", "yr": "year", "sem": "semester",
    "dept": "department", "admsn": "admission", "cresnt": "crescent", "uni": "university",
    "clg": "college", "sch": "school", "info": "information", "d": "the"
}

# Text normalization
def normalize_text(text):
    text = re.sub(r'([^a-zA-Z0-9\s])', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    return text

def preprocess_text(text):
    text = normalize_text(text)
    words = text.split()
    expanded = [abbreviations.get(word.lower(), word) for word in words]
    corrected = []
    for word in expanded:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        corrected.append(suggestions[0].term if suggestions else word)
    return ' '.join(corrected)

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load Q&A data
@st.cache_resource
def load_data():
    with open("qa_dataset.json", "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
    df = pd.DataFrame(qa_pairs)
    return df

# Find best match response
def find_response(user_input, dataset, question_embeddings, model, threshold=0.4):
    original_input = user_input
    user_input = preprocess_text(user_input)

    greetings = [
        "hi", "hello", "hey", "hi there", "greetings", "how are you",
        "how are you doing", "how's it going", "can we talk?",
        "can we have a conversation?", "okay", "i'm fine", "i am fine"
    ]
    if any(user_input.lower().startswith(greet) for greet in greetings):
        return random.choice([
            "Hello!", "Hi there!", "Hey!", "Greetings!",
            "I'm doing well, thank you!", "Sure pal", "Okay"
        ])

    # Direct course code match
    code_match = re.search(r"\b([A-Z]{2,}-?\d{3})\b", original_input.upper())
    if code_match:
        code = code_match.group(1)
        exact_match = dataset[dataset['question'].str.contains(code, case=False, regex=False)]
        if not exact_match.empty:
            response = exact_match.iloc[0]['answer']
            department = exact_match.iloc[0].get("department", "Unknown")
            response += f"\n\n_(Department: {department})_"
            return response

    # Semantic match
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]
    top_score = torch.max(cos_scores).item()
    top_index = torch.argmax(cos_scores).item()

    if top_score < threshold:
        return random.choice([
            "I'm sorry, I don't understand your question.",
            "Can you rephrase your question?"
        ])
    response = dataset.iloc[top_index]['answer']
    department = dataset.iloc[top_index].get("department", "Unknown")

    if random.random() < 0.2:
        uncertainty_phrases = [
            "I think ", "Maybe this helps: ", "Here's what I found: ",
            "Possibly: ", "It could be: "
        ]
        response = random.choice(uncertainty_phrases) + response
    return response

# Streamlit UI
st.set_page_config(page_title="🎓 Crescent University Chatbot", page_icon="🎓")
st.title("🎓 Crescent University Chatbot")

# Load resources
model = load_model()
dataset = load_data()
question_embeddings = model.encode(dataset['question'].tolist(), convert_to_tensor=True)

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar: clear chat
with st.sidebar:
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []

# Display previous chat
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask me anything about Crescent University...")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    response = find_response(user_input, dataset, question_embeddings, model)

    with st.chat_message("assistant"):
        st.markdown(response)
        
    st.session_state.chat_history.append({"role": "assistant", "content": response})
