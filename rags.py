# --- Imports ---
import streamlit as st
import re
import time
import json
import random
import os
import pkg_resources
from symspellpy.symspellpy import SymSpell
from sentence_transformers import SentenceTransformer, util
from openai.error import AuthenticationError
import openai

# --- Constants ---
ABBREVIATIONS = {
    "u": "you", "r": "are", "ur": "your", "cn": "can", "cud": "could",
    "shud": "should", "wud": "would", "abt": "about", "bcz": "because",
    "plz": "please", "pls": "please", "tmrw": "tomorrow", "wat": "what",
    "wats": "what is", "info": "information", "yr": "year", "sem": "semester",
    "admsn": "admission", "clg": "college", "sch": "school", "uni": "university",
    "cresnt": "crescent", "l": "level", "d": "the", "msg": "message",
    "idk": "i don't know", "imo": "in my opinion", "asap": "as soon as possible",
    "dept": "department", "reg": "registration", "fee": "fees", "pg": "postgraduate",
    "app": "application", "req": "requirement", "nd": "national diploma",
    "a-level": "advanced level", "alevel": "advanced level", "2nd": "second",
    "1st": "first", "nxt": "next", "prev": "previous", "exp": "experience",
    "CSC": "department of Computer Science", "Mass comm": "department of Mass Communication",
    "law": "department of law", "Acc": "department of Accounting"
}

SYNONYMS = {
    "lecturers": "academic staff", "professors": "academic staff",
    "teachers": "academic staff", "instructors": "academic staff",
    "tutors": "academic staff", "staff members": "staff",
    "head": "dean", "hod": "head of department", "dept": "department",
    "school": "university", "college": "faculty", "course": "subject",
    "class": "course", "subject": "course", "unit": "credit",
    "credit unit": "unit", "course load": "unit", "non teaching": "non-academic",
    "admin worker": "non-academic staff", "support staff": "non-academic staff",
    "clerk": "non-academic staff", "receptionist": "non-academic staff",
    "secretary": "non-academic staff", "tech staff": "technical staff",
    "hostel": "accommodation", "lodging": "accommodation", "room": "accommodation",
    "school fees": "tuition", "acceptance fee": "admission fee", "fees": "tuition",
    "enrol": "apply", "join": "apply", "sign up": "apply", "admit": "apply",
    "requirement": "criteria", "conditions": "criteria", "needed": "required",
    "needed for": "required for", "who handles": "who manages"
}

ABUSE_WORDS = ["fuck", "shit", "bitch", "nigga", "dumb", "sex"]
ABUSE_PATTERN = re.compile(r'\b(' + '|'.join(map(re.escape, ABUSE_WORDS)) + r')\b', re.IGNORECASE)

DEPARTMENT_NAMES = [d.lower() for d in [
    "Computer Science", "Mass Communication", "Law", "Microbiology",
    "Accounting", "Political Science", "Business Administration", "Business Admin"
]]

LONG_TERM_MEMORY_PATH = "long_term_memory.json"

# --- Long-Term Memory Load/Save ---
def load_long_term_memory():
    if os.path.exists(LONG_TERM_MEMORY_PATH):
        with open(LONG_TERM_MEMORY_PATH, "r") as f:
            return json.load(f)
    else:
        return {}

def save_long_term_memory(memory):
    with open(LONG_TERM_MEMORY_PATH, "w") as f:
        json.dump(memory, f, indent=2)

# --- Cache Loaders ---
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

# --- Text Normalization ---
def normalize_text(text, sym_spell):
    text = text.lower()
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    corrected = suggestions[0].term if suggestions else text
    for abbr, full in ABBREVIATIONS.items():
        corrected = re.sub(rf'\b{re.escape(abbr)}\b', full, corrected)
    for syn, rep in SYNONYMS.items():
        corrected = re.sub(rf'\b{re.escape(syn)}\b', rep, corrected)
    return corrected

# --- Utility: Greetings ---
def is_greeting(text):
    return text.lower().strip() in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]

def get_random_greeting_response():
    return random.choice([
        "Hello! How can I assist you today?",
        "Hi there! What can I help you with?",
        "Hey! Feel free to ask me anything about Crescent University.",
        "Greetings! How may I be of service?",
        "Hello! Ready to help you with any questions."
    ])

def is_farewell(text):
    return text.lower().strip() in ["bye", "goodbye", "see you", "later", "farewell", "cya", "peace", "exit"]

def get_random_farewell_response():
    return random.choice([
        "Goodbye! Have a great day!",
        "See you later! Feel free to come back anytime.",
        "Bye! Take care!",
        "Farewell! Let me know if you need anything else.",
        "Peace out! Hope to chat again soon."
    ])

# --- Memory & Follow-up ---
def update_chat_memory(norm_input, memory):
    for dept in DEPARTMENT_NAMES:
        if re.search(rf"\b{re.escape(dept)}\b", norm_input):
            memory["department"] = dept.title()
            break
    dep_match = re.search(r"department of ([a-zA-Z &]+)", norm_input)
    if dep_match:
        memory["department"] = dep_match.group(1).title()
    lvl_match = re.search(r"(100|200|300|400|500)\s*level", norm_input)
    if lvl_match:
        memory["level"] = lvl_match.group(1)
    topic_keywords = {
        "admission": ["admission", "apply", "jamb", "requirement"],
        "fees": ["fee", "tuition", "cost", "school fees"],
        "courses": ["course", "subject", "unit", "curriculum", "study"],
        "accommodation": ["accommodation", "hostel", "reside", "lodging"],
        "graduation": ["graduation", "convocation"],
        "exam": ["exam", "test", "cgpa", "grade"],
        "scholarship": ["scholarship", "aid", "bursary"],
        "dress code": ["dress code", "uniform", "appearance"]
    }
    for topic, kws in topic_keywords.items():
        if any(kw in norm_input for kw in kws):
            memory["topic"] = topic
            break
    return memory

def resolve_follow_up(raw_input, memory):
    text = raw_input.strip().lower()
    m = re.match(r"what about (\d{3}) level", text)
    if m and memory.get("department"):
        return f"What are the {m.group(1)} level courses in {memory['department']}?"
    if text.startswith("what about") and memory.get("level") and memory.get("department"):
        return f"What are the {memory['level']} level courses in {memory['department']}?"
    m2 = re.match(r"do they also .* in ([a-zA-Z &]+)\?", text)
    if m2 and memory.get("topic"):
        return f"Do they also offer {memory['topic']} in {m2.group(1).title()}?"
    m3 = re.match(r"how about .* for ([a-zA-Z &]+)\?", text)
    if m3 and memory.get("topic"):
        return f"Do they also offer {memory['topic']} in {m3.group(1).title()}?"
    return raw_input

# --- Retrieval & Fallback ---
def retrieve_answer(user_input, dataset, q_embeds, embed_model):
    for item in dataset:
        if user_input.strip().lower() in item["question"].strip().lower():
            return item["answer"], 1.0
    user_embed = embed_model.encode(user_input, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.pytorch_cos_sim(user_embed, q_embeds)[0]
    best_idx = scores.argmax().item()
    return dataset[best_idx]["answer"], float(scores[best_idx])

def build_contextual_prompt(messages, short_term_memory, long_term_memory, user_key="global", max_turns=6):
    recent = messages[-max_turns * 2:]
    long_mem_user = long_term_memory.get(user_key, {"departments": [], "topics": [], "levels": []})

    mem_info = (
        f"- Department: {short_term_memory.get('department') or 'unspecified'}\n"
        f"- Level: {short_term_memory.get('level') or 'unspecified'}\n"
        f"- Topic: {short_term_memory.get('topic') or 'unspecified'}\n"
        f"Long-term memory summary:\n"
        f"- Past departments discussed: {', '.join(long_mem_user.get('departments', [])) or 'none'}\n"
        f"- Past topics discussed: {', '.join(long_mem_user.get('topics', [])) or 'none'}\n"
        f"- Past levels discussed: {', '.join(long_mem_user.get('levels', [])) or 'none'}"
    )

    system_msg = {
        "role": "system",
        "content": "You are a helpful assistant for Crescent University.\n" + mem_info,
    }
    return [system_msg] + recent

def call_gpt_api(prompt_messages, model="gpt-4o-mini", temperature=0.3):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=prompt_messages,
            temperature=temperature,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()
    except AuthenticationError:
        return "OpenAI API key is invalid or missing."
    except Exception as e:
        return f"Error contacting GPT API: {e}"

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Crescent University Assistant", page_icon="🎓")
    st.title("🎓 Crescent University Chatbot Assistant")

    # Load or initialize resources
    if "embed_model" not in st.session_state:
        embed_model, sym_spell, dataset, q_embeds = load_all_data()
        st.session_state.embed_model = embed_model
        st.session_state.sym_spell = sym_spell
        st.session_state.dataset = dataset
        st.session_state.q_embeds = q_embeds
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your Crescent University assistant. Ask me anything!"}]
        st.session_state.short_term_memory = {"department": None, "topic": None, "level": None}
    if "long_term_memory" not in st.session_state:
        st.session_state.long_term_memory = load_long_term_memory()

    # Display chat messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Bot:** {msg['content']}")

    # User input box
    user_input = st.text_input("You:", key="input", placeholder="Type your question here and press Enter...")

    if user_input:
        # Reset input box
        st.session_state.input = ""

        # Check abuse words
        if ABUSE_PATTERN.search(user_input):
            response = "Please avoid using inappropriate language. How can I assist you properly?"
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.experimental_rerun()

        # Greetings / Farewell
        if is_greeting(user_input):
            response = get_random_greeting_response()
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.experimental_rerun()

        if is_farewell(user_input):
            response = get_random_farewell_response()
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.experimental_rerun()

        # Normalize and resolve follow-up
        norm_input = normalize_text(user_input, st.session_state.sym_spell)
        norm_input = resolve_follow_up(norm_input, st.session_state.short_term_memory)

        # Update short-term memory
        st.session_state.short_term_memory = update_chat_memory(norm_input, st.session_state.short_term_memory)

        # Update long-term memory
        memory = st.session_state.long_term_memory
        user_key = "global"  # can extend to multi-user if needed

        if user_key not in memory:
            memory[user_key] = {"departments": [], "topics": [], "levels": []}

        dep = st.session_state.short_term_memory.get("department")
        if dep and dep.lower() not in [d.lower() for d in memory[user_key]["departments"]]:
            memory[user_key]["departments"].append(dep)

        topic = st.session_state.short_term_memory.get("topic")
        if topic and topic.lower() not in [t.lower() for t in memory[user_key]["topics"]]:
            memory[user_key]["topics"].append(topic)

        lvl = st.session_state.short_term_memory.get("level")
        if lvl and lvl not in memory[user_key]["levels"]:
            memory[user_key]["levels"].append(lvl)

        save_long_term_memory(memory)

        st.session_state.messages.append({"role": "user", "content": user_input})

        # Try exact or semantic retrieval
        answer, score = retrieve_answer(norm_input, st.session_state.dataset, st.session_state.q_embeds, st.session_state.embed_model)

        if score >= 0.55:
            bot_response = answer
        else:
            # GPT fallback with context
            prompt_messages = build_contextual_prompt(st.session_state.messages, st.session_state.short_term_memory, st.session_state.long_term_memory, user_key=user_key)
            bot_response = call_gpt_api(prompt_messages)

        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        st.experimental_rerun()

if __name__ == "__main__":
    main()
