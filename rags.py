import json
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from symspellpy.symspellpy import SymSpell, Verbosity
import numpy as np
import os
import re
import random
import pkg_resources
from dotenv import load_dotenv
import openai

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables")

openai.api_key = openai_api_key

# Test OpenAI connection (optional)
# response = openai.ChatCompletion.create(
#     model="gpt-4o-mini",
#     messages=[{"role": "user", "content": "Hello"}]
# )
# print(response['choices'][0]['message']['content'])

# Load upgraded sentence transformer model for better semantic understanding
model = SentenceTransformer('all-mpnet-base-v2', device='cpu')

# Initialize SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Abbreviation mapping
abbreviations = {
    "u": "you", "r": "are", "ur": "your", "ow": "how", "pls": "please", "plz": "please",
    "tmrw": "tomorrow", "cn": "can", "wat": "what", "cud": "could", "shud": "should",
    "wud": "would", "abt": "about", "bcz": "because", "btw": "between", "asap": "as soon as possible",
    "idk": "i don't know", "imo": "in my opinion", "msg": "message", "doc": "document", "d": "the",
    "yr": "year", "sem": "semester", "dept": "department", "admsn": "admission",
    "cresnt": "crescent", "uni": "university", "clg": "college", "sch": "school",
    "info": "information", "l": "level", "CSC": "Computer Science", "ECO": "Economics with Operations Research",
    "PHY": "Physics", "STAT": "Statistics", "1st": "First", "2nd": "Second", "tech staff": "technical staff",
    "it people": "technical staff", "lab helper": "technical staff", "computer staff": "technical staff",
    "equipment handler": "technical staff", "office staff": "non-academic staff", "admin worker": "non-academic staff",
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

# Text preprocessing
def normalize_text(text):
    text = re.sub(r'([^a-zA-Z0-9\s])', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    return text

def preprocess_text(text):
    text = normalize_text(text.lower())

    # Phrase-level replacements (multi-word terms)
    for phrase, replacement in {**abbreviations, **synonym_map}.items():
        if phrase in text:
            text = text.replace(phrase, replacement)

    words = text.split()
    expanded = []
    for word in words:
        word = abbreviations.get(word, word)  # Still check for single-word replacements
        word = synonym_map.get(word, word)
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        corrected = suggestions[0].term if suggestions else word
        expanded.append(corrected)

    return ' '.join(expanded)

# Load dataset
with open("qa_dataset.json") as f:
    data = json.load(f)

processed_questions = [preprocess_text(item['question']) for item in data]
question_embeddings = model.encode(processed_questions, convert_to_tensor=True)

# Find top-k similar questions from dataset
def find_top_k_matches(user_input, dataset, embeddings, top_k=3):
    cleaned = preprocess_text(user_input)
    user_embedding = model.encode(cleaned, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(user_embedding, embeddings)[0]
    top_k_indices = np.argsort(-similarities.cpu().numpy())[:top_k]
    top_k_matches = []
    for idx in top_k_indices:
        top_k_matches.append({
            "question": dataset[idx]['question'],
            "answer": dataset[idx]['answer'],
            "score": float(similarities[idx])
        })
    return top_k_matches

# GPT fallback with RAG-style context using classic openai SDK
def gpt_fallback_with_context(user_input, top_matches):
    context_text = "\n".join(
        [f"{i+1}. Q: {item['question']}\n   A: {item['answer']}" for i, item in enumerate(top_matches)]
    )
    prompt = (
        f"You are a helpful chatbot for Crescent University. Use the following context to answer the user's question.\n\n"
        f"Context:\n{context_text}\n\n"
        f"User Question: {user_input}"
    )

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful chatbot for Crescent University."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content'].strip()

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
    cleaned = re.sub(r"[^\w\s]", "", text.lower().strip())
    return cleaned in greeting_inputs

# UI chat bubbles
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

# Streamlit app
st.title("🎓 Crescent University Chatbot")
st.markdown("Ask me anything about Crescent University!")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Your question:", key="input")

if user_input:
    with st.spinner("Thinking..."):
        if is_greeting(user_input):
            final_response = random.choice(greeting_responses)
        else:
            top_matches = find_top_k_matches(user_input, data, question_embeddings, top_k=3)
            best_match = top_matches[0]
            if best_match['score'] >= 0.75:  # You can fine-tune this threshold
                final_response = best_match['answer']
            else:
                try:
                    final_response = gpt_fallback_with_context(user_input, top_matches)
                except Exception as e:
                    final_response = (
                        "I'm not sure how to answer that right now. "
                        "Please try rephrasing your question."
                    )

        st.session_state.history.append({
            "user": user_input,
            "bot": final_response
        })

# Display chat history
for chat in st.session_state.history:
    st.markdown(render_message(chat["user"], is_user=True), unsafe_allow_html=True)
    st.markdown(render_message(chat["bot"], is_user=False), unsafe_allow_html=True)
