import streamlit as st
import re
import time
import random
import json
import os
from sentence_transformers import SentenceTransformer, util
from symspellpy.symspellpy import SymSpell
from openai.error import AuthenticationError
import pkg_resources
import openai

# --------------------------
# Normalization dictionaries
ABBREVIATIONS = {
    "u": "you", "r": "are", "ur": "your", "cn": "can", "cud": "could", "shud": "should", "wud": "would",
    "abt": "about", "bcz": "because", "plz": "please", "pls": "please", "tmrw": "tomorrow", "wat": "what",
    "wats": "what is", "info": "information", "yr": "year", "sem": "semester", "admsn": "admission",
    "clg": "college", "sch": "school", "uni": "university", "cresnt": "crescent", "l": "level", 
    "d": "the", "msg": "message", "idk": "i don't know", "imo": "in my opinion", "asap": "as soon as possible",
    "dept": "department", "reg": "registration", "fee": "fees", "pg": "postgraduate", "app": "application",
    "req": "requirement", "nd": "national diploma", "a-level": "advanced level", "alevel": "advanced level",
    "2nd": "second", "1st": "first", "nxt": "next", "prev": "previous", "exp": "experience", "CSC": "department of Computer Science",
    "Mass comm": "department of Mass Communication", "law": "department of law", "Acc": "department of Accounting"
}

SYNONYMS = {
    "lecturers": "academic staff", "professors": "academic staff", "teachers": "academic staff",
    "instructors": "academic staff", "tutors": "academic staff", "staff members": "staff",
    "head": "dean", "hod": "head of department", "dept": "department", "school": "university",
    "college": "faculty", "course": "subject", "class": "course", "subject": "course", 
    "unit": "credit", "credit unit": "unit", "course load": "unit", "non teaching": "non-academic",
    "admin worker": "non-academic staff", "support staff": "non-academic staff", "clerk": "non-academic staff",
    "receptionist": "non-academic staff", "secretary": "non-academic staff", "tech staff": "technical staff",
    "hostel": "accommodation", "lodging": "accommodation", "room": "accommodation", "school fees": "tuition",
    "acceptance fee": "admission fee", "fees": "tuition", "enrol": "apply", "join": "apply", 
    "sign up": "apply", "admit": "apply", "requirement": "criteria", "conditions": "criteria",
    "needed": "required", "needed for": "required for", "who handles": "who manages"
}

ABUSE_WORDS = ["fuck", "shit", "bitch", "nigga", "dumb", "sex"]
ABUSE_PATTERN = re.compile(r'\b(' + '|'.join(map(re.escape, ABUSE_WORDS)) + r')\b', re.IGNORECASE)

MEMORY_FILE = "long_term_memory.json"

# --------------------------
@st.cache_resource
def init_symspell():
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dict_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
    return sym_spell

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_dataset():
    with open("qa_dataset.json", "r") as f:
        return json.load(f)

@st.cache_resource
def load_all_data():
    embed_model = load_embedding_model()
    sym_spell = init_symspell()
    dataset = load_dataset()
    questions = [item["question"] for item in dataset]
    q_embeds = embed_model.encode(questions, convert_to_tensor=True, normalize_embeddings=True)
    return embed_model, sym_spell, dataset, q_embeds

@st.cache_resource
def load_long_term_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    else:
        return {"student_name": None, "faculty": None, "location": None, "department": None, "level": None, "topic": None}

def save_long_term_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f)

def normalize_text(text, sym_spell):
    text = text.lower()
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    corrected = suggestions[0].term if suggestions else text

    for abbr, full in ABBREVIATIONS.items():
        corrected = re.sub(rf'\b{re.escape(abbr)}\b', full, corrected)
    for syn, rep in SYNONYMS.items():
        corrected = re.sub(rf'\b{re.escape(syn)}\b', rep, corrected)
    return corrected

def is_greeting(text):
    return any(text.lower().strip() == g for g in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"])

def get_random_greeting_response():
    return random.choice([
        "Hello! How can I assist you today?",
        "Hi there! What can I help you with?",
        "Hey! Feel free to ask me anything about Crescent University.",
        "Greetings! How may I be of service?",
        "Hello! Ready to help you with any questions."
    ])

def is_farewell(text):
    return any(text.lower().strip() == f for f in ["bye", "goodbye", "see you", "later", "farewell", "cya", "peace", "exit"])

def get_random_farewell_response():
    return random.choice([
        "Goodbye! Have a great day!",
        "See you later! Feel free to come back anytime.",
        "Bye! Take care!",
        "Farewell! Let me know if you need anything else.",
        "Peace out! Hope to chat again soon."
    ])

def retrieve_answer(user_input, dataset, q_embeds, embed_model):
    # Try exact match
    for item in dataset:
        if user_input.strip().lower() in item["question"].strip().lower():
            return item["question"], item["answer"], 1.0

    # Semantic match
    user_embed = embed_model.encode(user_input, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.pytorch_cos_sim(user_embed, q_embeds)[0]
    best_score = float(scores.max())
    best_idx = int(scores.argmax())
    return dataset[best_idx]["question"], dataset[best_idx]["answer"], best_score

def build_contextual_prompt(messages, new_input, max_turns=3):
    recent = messages[-max_turns * 2:] if len(messages) >= 2 else []
    chat = [{"role": m["role"], "content": m["content"]} for m in recent]
    chat.append({"role": "user", "content": new_input})
    return [{"role": "system", "content": "You are a helpful assistant for Crescent University."}] + chat

def update_memory(user_input):
    memory = st.session_state.memory
    # Try to extract name
    if "my name is" in user_input:
        name_part = user_input.split("my name is")[-1].strip()
        if name_part:
            memory["student_name"] = name_part.split()[0].capitalize()
    # Try to extract faculty
    if "i am in" in user_input and "faculty" in user_input:
        fac_part = user_input.split("i am in")[-1].strip()
        if "faculty" in fac_part:
            fac_name = fac_part.split("faculty")[0].strip()
            memory["faculty"] = fac_name.capitalize()
    # Try to extract location
    if "i live in" in user_input:
        loc_part = user_input.split("i live in")[-1].strip()
        if loc_part:
            memory["location"] = loc_part.capitalize()
    save_long_term_memory(memory)

def check_memory_question(user_input):
    memory = st.session_state.memory
    if ("what is my name" in user_input or "how about my name" in user_input) and memory["student_name"]:
        return f"Your name is {memory['student_name']}."
    elif ("how about my faculty" in user_input or "what is my faculty" in user_input) and memory["faculty"]:
        return f"You are in the {memory['faculty']} Faculty."
    elif ("what is my location" in user_input or "how about my location" in user_input) and memory["location"]:
        return f"You are located in {memory['location']}."
    return None

def main():
    st.set_page_config(page_title="Crescent University Chatbot", page_icon="🎓")
    st.title("🎓 Crescent University Chatbot")
    st.markdown("Ask me anything about your department, courses, or the university.")

    openai_api_key = st.secrets.get("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OpenAI API key not configured. GPT fallback disabled.")
        openai_enabled = False
    else:
        openai.api_key = openai_api_key
        openai_enabled = True

    if "embed_model" not in st.session_state:
        embed_model, sym_spell, dataset, q_embeds = load_all_data()
        st.session_state.embed_model = embed_model
        st.session_state.sym_spell = sym_spell
        st.session_state.dataset = dataset
        st.session_state.q_embeds = q_embeds
        st.session_state.openai_enabled = openai_enabled

    if "memory" not in st.session_state:
        st.session_state.memory = load_long_term_memory()

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your Crescent University assistant. Ask me anything!"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Type your question here...")

    if user_input:
        norm_input = normalize_text(user_input, st.session_state.sym_spell)

        if ABUSE_PATTERN.search(norm_input):
            response = "Sorry, I can’t help with that. Try asking about something academic."

        elif is_greeting(norm_input):
            response = get_random_greeting_response()

        elif is_farewell(norm_input):
            response = get_random_farewell_response()

        else:
            # Update and check memory first
            update_memory(norm_input)
            memory_response = check_memory_question(norm_input)

            if memory_response:
                response = memory_response

            else:
                matched_q, answer, score = retrieve_answer(
                    user_input,
                    st.session_state.dataset,
                    st.session_state.q_embeds,
                    st.session_state.embed_model,
                )
                threshold = 0.65
                if score < threshold and st.session_state.openai_enabled:
                    # Fallback to OpenAI GPT-3.5 Turbo
                    try:
                        messages = build_contextual_prompt(st.session_state.messages, user_input)
                        chat_completion = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=messages,
                            temperature=0.2,
                            max_tokens=300,
                        )
                        response = chat_completion.choices[0].message.content.strip()
                    except AuthenticationError:
                        response = "OpenAI API key invalid or quota exceeded. Please try again later."
                elif score >= threshold:
                    response = answer
                else:
                    response = "Sorry, I couldn't find a good answer to your question."

        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": response})

        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            st.markdown(response)

        st.experimental_rerun()

if __name__ == "__main__":
    main()
