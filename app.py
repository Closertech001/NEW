
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

# Load your dataset
input_file = "qa_dataset.json"  # Replace with your actual filename
output_file = "qa_dataset_with_departments.json"

# Department mapping
department_map = {
    "GST": "General Studies",
    "MTH": "Mathematics",
    "PHY": "Physics",
    "STA": "Statistics",
    "COS": "Computer Science",
    "CUAB-CSC": "Computer Science",
    "CSC": "Computer Science",
    "IFT": "Computer Science",
    "SEN": "Software Engineering",
    "ENT": "Entrepreneurship",
    "CYB": "Cybersecurity",
    "ICT": "Information and Communication Technology",
    "DTS": "Data Science",
    "CUAB-CPS": "Computer Science",
    "CUAB-ECO": "Economics with Operations Research",
    "ECO": "Economics with Operations Research",
    "SSC": "Social Sciences",
    "CUAB-BCO": "Economics with Operations Research",
    "LIB": "Library Studies",
    "LAW": "Law (BACOLAW)",
    "GNS": "General Studies",
    "ENG": "English",
    "SOS": "Sociology",
    "PIS": "Political Science",
    "CPS": "Computer Science",
    "LPI": "Law (BACOLAW)",
    "ICL": "Law (BACOLAW)",
    "LPB": "Law (BACOLAW)",
    "TPT": "Law (BACOLAW)",
    "FAC": "Agricultural Sciences",
    "ANA": "Anatomy",
    "BIO": "Biological Sciences",
    "CHM": "Chemical Sciences",
    "CUAB-BCH": "Biochemistry",
    "CUAB": "Crescent University - General"
}

def extract_prefix(code):
    match = re.match(r"([A-Z\-]+)", code)
    return match.group(1) if match else None

def main():
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        match = re.search(r"What course is ([A-Z\-0-9]+)", entry.get("question", ""))
        if match:
            code = match.group(1)
            prefix = extract_prefix(code)
            department = department_map.get(prefix, "Unknown")
            entry["department"] = department

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Updated dataset saved to {output_file}")

if __name__ == "__main__":
    main()

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

# Normalize and preprocess
def normalize_text(text):
    # Preserve uppercase acronyms like GNS, GST, PHY etc.
    text = re.sub(r'([^a-zA-Z0-9\s])', '', text)  # Remove symbols
    text = re.sub(r'(.)\1{2,}', r'\1', text)  # Fix repeated chars
    return text  # Avoid forcing lowercasing

def preprocess_text(text):
    text = normalize_text(text)
    words = text.split()
    expanded = [abbreviations.get(word, word) for word in words]
    corrected = []
    for word in expanded:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        corrected.append(suggestions[0].term if suggestions else word)
    return ' '.join(corrected)

# Load the model once
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load Q&A Dataset
@st.cache_resource
def load_data():
    with open("qa_dataset.json", "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
    df = pd.DataFrame(qa_pairs)
    return df

# Find best match response
def find_response(user_input, dataset, question_embeddings, model, threshold=0.4):
    user_input = preprocess_text(user_input)
    greetings = [
        "hi", "hello", "hey", "hi there", "greetings", "how are you",
        "how are you doing", "how's it going", "can we talk?",
        "can we have a conversation?", "okay", "i'm fine", "i am fine"
    ]
    if user_input in greetings:
        return random.choice([
            "Hello!", "Hi there!", "Hey!", "Greetings!",
            "I'm doing well, thank you!", "Sure pal", "Okay"
        ])

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
    if random.random() < 0.2:
        uncertainty_phrases = [
            "I think ", "Maybe this helps: ", "Here's what I found: ",
            "Possibly: ", "It could be: "
        ]
        response = random.choice(uncertainty_phrases) + response
    return response

# Initialize
st.set_page_config(page_title="🎓 Crescent University Chatbot", page_icon="🎓")
st.title("🎓 Crescent University Chatbot")

# Load resources
model = load_model()
dataset = load_data()
question_embeddings = model.encode(dataset['question'].tolist(), convert_to_tensor=True)

# Session state chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar clear button
with st.sidebar:
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []

# Chat interface
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about Crescent University..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    response = find_response(prompt, dataset, question_embeddings, model)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})
