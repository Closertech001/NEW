import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import random
import re
from symspellpy.symspellpy import SymSpell, Verbosity
import pkg_resources
import openai

# Load OpenAI API key from secrets
openai_api_key = "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"
openai.api_key = st.secrets.toml["openai_api_key"]

# Load SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Abbreviations dictionary
abbreviations = {
    "u": "you", "r": "are", "ur": "your", "ow": "how", "pls": "please", "plz": "please",
    "tmrw": "tomorrow", "cn": "can", "wat": "what", "cud": "could", "shud": "should",
    "wud": "would", "abt": "about", "bcz": "because", "bcoz": "because", "btw": "between",
    "asap": "as soon as possible", "idk": "i don't know", "imo": "in my opinion",
    "msg": "message", "doc": "document", "d": "the", "yr": "year", "sem": "semester",
    "dept": "department", "admsn": "admission", "cresnt": "crescent", "uni": "university",
    "clg": "college", "sch": "school", "info": "information"
}

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    return text

def preprocess_text(text):
    text = normalize_text(text)
    words = text.split()
    expanded = [abbreviations.get(word, word) for word in words]
    corrected = []
    for word in expanded:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        corrected.append(suggestions[0].term if suggestions else word)
    return ' '.join(corrected)

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    qa_pairs = []
    with open("UNIVERSITY DATASET.txt", 'r', encoding='utf-8') as file:
        question, answer = None, None
        for line in file:
            line = line.strip()
            if line.startswith("Q:"):
                question = line[2:].strip()
            elif line.startswith("A:"):
                answer = line[2:].strip()
                if question and answer:
                    qa_pairs.append((question, answer))
                    question, answer = None, None
    return pd.DataFrame(qa_pairs, columns=["question", "response"])

def find_response(user_input, dataset, question_embeddings, model, threshold=0.6):
    processed_input = preprocess_text(user_input)

    greetings = [
        "hi", "hello", "hey", "hi there", "greetings", "how are you",
        "how are you doing", "how's it going", "can we talk?",
        "can we have a conversation?", "okay", "i'm fine", "i am fine"
    ]
    if processed_input in greetings:
        return random.choice([
            "Hello!", "Hi there!", "Hey!", "Greetings!",
            "I'm doing well, thank you!", "Sure pal", "Okay"
        ])

    user_embedding = model.encode(processed_input, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]
    top_score = torch.max(cos_scores).item()
    top_index = torch.argmax(cos_scores).item()

    if top_score < threshold:
        return random.choice([
            "I'm sorry, I don't understand your question.",
            "Can you rephrase your question?"
        ])

    response = dataset.iloc[top_index]['response']
    if random.random() < 0.2:
        uncertainty_phrases = [
            "I think ", "Maybe this helps: ", "Here's what I found: ",
            "Possibly: ", "It could be: "
        ]
        response = random.choice(uncertainty_phrases) + response
    return response

def gpt_response_with_memory(chat_history, current_input):
    messages = [
        {"role": "system", "content": "You are a helpful assistant for Crescent University. Use past conversation context to answer clearly and politely."}
    ]
    for chat in chat_history:
        messages.append({"role": chat["role"], "content": chat["content"]})
    messages.append({"role": "user", "content": current_input})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or "gpt-3.5-turbo"
            messages=messages,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return "Sorry, there was an error using the smart assistant. Please try again later."

st.set_page_config(page_title="🎓 Crescent University Chatbot", page_icon="🎓")
st.title("🎓 Crescent University Chatbot")

model = load_model()
dataset = load_data()
question_embeddings = model.encode(dataset['question'].tolist(), convert_to_tensor=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything about Crescent University..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    bert_response = find_response(prompt, dataset, question_embeddings, model)

    fallback_phrases = ["i'm sorry", "can you rephrase", "i don't understand"]
    if any(p in bert_response.lower() for p in fallback_phrases):
        final_response = gpt_response_with_memory(st.session_state.chat_history, prompt)
    else:
        final_response = bert_response

    with st.chat_message("assistant"):
        st.markdown(final_response)
    st.session_state.chat_history.append({"role": "assistant", "content": final_response})
