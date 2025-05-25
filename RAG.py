# Import modules
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import random
import re
from symspellpy.symspellpy import SymSpell, Verbosity
import pkg_resources
import json
import openai
import os

# Setup
openai.api_key = os.getenv("OPENAI_API_KEY")

# SymSpell for spell correction
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Abbreviations mapping
abbreviations = {"u": "you", "r": "are", "ur": "your", "ow": "how", "pls": "please", "plz": "please", "tmrw": "tomorrow", "cn": "can", "wat": "what", "cud": "could", "shud": "should", "wud": "would", "abt": "about", "bcz": "because", "bcoz": "because", "btw": "between", "asap": "as soon as possible", "idk": "i don't know", "imo": "in my opinion", "msg": "message", "doc": "document", "d": "the", "yr": "year", "sem": "semester", "dept": "department", "admsn": "admission", "cresnt": "crescent", "uni": "university", "clg": "college", "sch": "school", "info": "information", "l": "level", "CSC": "Computer Science", "ECO": "Economics with Operations Research", "PHY": "Physics", "STAT": "Statistics", "1st": "First", "2nd": "Second"}

# Department mapping
department_map = {"GST": "General Studies", "MTH": "Mathematics", "PHY": "Physics", "STA": "Statistics", "COS": "Computer Science", "CUAB-CSC": "Computer Science", "CSC": "Computer Science", "IFT": "Computer Science", "SEN": "Software Engineering", "ENT": "Entrepreneurship", "CYB": "Cybersecurity", "ICT": "Information and Communication Technology", "DTS": "Data Science", "CUAB-CPS": "Computer Science", "CUAB-ECO": "Economics with Operations Research", "ECO": "Economics with Operations Research", "SSC": "Social Sciences", "CUAB-BCO": "Economics with Operations Research", "LIB": "Library Studies", "LAW": "Law (BACOLAW)", "GNS": "General Studies", "ENG": "English", "SOS": "Sociology", "PIS": "Political Science", "CPS": "Computer Science", "LPI": "Law (BACOLAW)", "ICL": "Law (BACOLAW)", "LPB": "Law (BACOLAW)", "TPT": "Law (BACOLAW)", "FAC": "Agricultural Sciences", "ANA": "Anatomy", "BIO": "Biological Sciences", "CHM": "Chemical Sciences", "CUAB-BCH": "Biochemistry", "CUAB": "Crescent University - General"}

# Text processing
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

def extract_prefix(code):
    match = re.match(r"([A-Z\-]+)", code)
    return match.group(1) if match else None

# Load resources
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_data():
    with open("qa_dataset.json", "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
    return pd.DataFrame(qa_pairs)

@st.cache_data
def compute_question_embeddings(questions: list):
    model = load_model()
    return model.encode(questions, convert_to_tensor=True)

# Build RAG context string
def build_context_string(dataset, top_indices):
    context = ""
    for idx in top_indices:
        row = dataset.iloc[idx.item()]
        context += f"Q: {row['question']}\nA: {row['answer']}\n\n"
    return context.strip()

# GPT-4 response

def fallback_openai(user_input, context_string=None):
    system_prompt = (
        "You are a helpful assistant for Crescent University. "
        "Use the provided information to answer the user's question accurately. "
        "If the answer is not clearly stated, say you don't know and suggest referring to official sources."
    )

    messages = [{"role": "system", "content": system_prompt}]
    user_prompt = f"Based on the following context:\n\n{context_string}\n\nAnswer the following question:\n{user_input}" if context_string else user_input
    messages.append({"role": "user", "content": user_prompt})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.3
        )
        return response.choices[0].message["content"].strip()
    except Exception:
        return "Sorry, I couldn't reach the server. Try again later."

# Main search + RAG logic
def find_response(user_input, dataset, embeddings, threshold=0.4):
    model = load_model()
    user_input_clean = preprocess_text(user_input)

    greetings = ["hi", "hello", "hey", "hi there", "greetings", "how are you", "how are you doing", "how's it going", "can we talk?", "can we have a conversation?", "okay", "i'm fine", "i am fine"]
    if user_input_clean.lower() in greetings:
        return random.choice(["Hello!", "Hi there!", "Hey!", "Greetings!","I'm doing well, thank you!", "Sure pal", "Okay", "I'm fine, thank you"]), None, 1.0, []

    user_embedding = model.encode(user_input_clean, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(user_embedding, embeddings)[0]
    top_scores, top_indices = torch.topk(cos_scores, k=5)

    top_score = top_scores[0].item()
    top_index = top_indices[0].item()

    context_string = build_context_string(dataset, top_indices)
    gpt_reply = fallback_openai(user_input, context_string)

    question = dataset.iloc[top_index]["question"]
    match = re.search(r"\b([A-Z]{2,}-?\d{3,})\b", question)
    department = None
    if match:
        code = match.group(1)
        prefix = extract_prefix(code)
        department = department_map.get(prefix, "Unknown")

    related_questions = [dataset.iloc[i.item()]["question"] for i in top_indices[1:]]
    return gpt_reply, department, top_score, related_questions
