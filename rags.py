import streamlit as st
import re
import time
import random
import json
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
    for item in dataset:
        if user_input.strip().lower() in item["question"].strip().lower():
            return item["question"], item["answer"], 1.0
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

def detect_topic(user_input):
    user_input_lower = user_input.lower()
    topic_keywords = [
        "100 level", "200 level", "300 level", "400 level",
        "law", "accounting", "computer science", "mass communication",
        "faculty", "department", "course", "semester", "admission",
        "fees", "registration", "tuition", "hostel", "accommodation",
        "location", "staff", "lecturer", "professor", "exam", "result"
    ]
    for keyword in topic_keywords:
        if keyword in user_input_lower:
            return keyword
    return None

def update_current_topic(user_input):
    detected = detect_topic(user_input)
    if detected:
        st.session_state.current_topic = detected

def combine_with_context(user_input):
    if len(user_input.split()) < 6 and st.session_state.get("current_topic"):
        return f"{st.session_state.current_topic} {user_input}"
    return user_input

def extract_user_info(text):
    info = {}
    name_match = re.search(r"\bmy name is (\w+)", text, re.IGNORECASE)
    if not name_match:
        name_match = re.search(r"\bi am ([A-Z][a-z]+)\b", text, re.IGNORECASE)
    if name_match:
        name_candidate = name_match.group(1).title()
        if name_candidate.lower() not in ["in", "on", "from", "at", "into", "under"]:
            info['name'] = name_candidate

    faculty_match = re.search(r"\b(i am|i'm) (a|an)? ?([\w\s]+) student\b", text, re.IGNORECASE)
    if faculty_match:
        info['faculty'] = faculty_match.group(3).strip().title()

    location_match = re.search(r"\b(from|located in|live in) ([\w\s]+)", text, re.IGNORECASE)
    if location_match:
        info['location'] = location_match.group(2).strip().title()

    return info

def personalize_response(response):
    if 'name' in st.session_state:
        response += f"\n\nNice to talk with you again, {st.session_state['name']}!"
    if 'faculty' in st.session_state:
        response += f"\nI remember you're in the {st.session_state['faculty']} department."
    if 'location' in st.session_state:
        response += f"\nAnd you're from {st.session_state['location']}, right?"
    return response

# --------------------------
def main():
    st.title("🎓 Crescent University Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_topic" not in st.session_state:
        st.session_state.current_topic = None

    embed_model, sym_spell, dataset, q_embeds = load_all_data()

    user_input = st.chat_input("Ask me anything about Crescent University...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        if is_greeting(user_input):
            bot_response = get_random_greeting_response()
        elif is_farewell(user_input):
            bot_response = get_random_farewell_response()
        elif ABUSE_PATTERN.search(user_input):
            bot_response = "Please avoid using abusive language."
        else:
            update_current_topic(user_input)
            expanded_input = combine_with_context(user_input)
            normalized_input = normalize_text(expanded_input, sym_spell)

            matched_question, matched_answer, confidence = retrieve_answer(normalized_input, dataset, q_embeds, embed_model)

            if confidence > 0.75:
                bot_response = matched_answer
            else:
                try:
                    openai.api_key = st.secrets["OPENAI_API_KEY"]
                    contextual_prompt = build_contextual_prompt(st.session_state.messages, normalized_input)
                    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=contextual_prompt)
                    bot_response = response.choices[0].message.content.strip()
                except AuthenticationError:
                    bot_response = "OpenAI key not configured. Please set your API key."
                except Exception as e:
                    bot_response = f"Something went wrong: {e}"

        # Extract and store user info
        user_info = extract_user_info(user_input)
        for key, value in user_info.items():
            st.session_state[key] = value

        bot_response = personalize_response(bot_response)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if __name__ == "__main__":
    main()
