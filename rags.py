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

# --- Utility: Greetings & Farewells ---
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
def get_top_k_relevant_snippets(user_input, dataset, embed_model, q_embeds, k=3):
    user_embed = embed_model.encode(user_input, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.pytorch_cos_sim(user_embed, q_embeds)[0]
    topk = scores.topk(k)
    snippets = []
    for idx in topk.indices:
        item = dataset[idx]
        snippets.append(f"Q: {item['question']}\nA: {item['answer']}")
    return "\n\n".join(snippets)

def summarize_memory(chat_history, max_turns=6):
    # Summarize last N turns in simple text, can be improved with GPT summarization if desired
    last_turns = chat_history[-max_turns*2:]  # each turn = user + assistant
    summary_lines = []
    for msg in last_turns:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"].replace("\n", " ")
        summary_lines.append(f"{role}: {content}")
    summary = "\n".join(summary_lines)
    return summary

def build_gpt_prompt(user_input, memory_summary, knowledge_snippets):
    system_msg = f"""You are a friendly, helpful, and professional assistant for Crescent University.
You have access to some past conversation context and university knowledge to help answer questions accurately.
    
Past conversation summary:
{memory_summary}

Relevant university knowledge:
{knowledge_snippets}

Please answer the user's question below clearly and politely:
"""
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_input}
    ]
    return messages

def is_abusive(text):
    return ABUSE_PATTERN.search(text) is not None

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="Crescent University Chatbot", page_icon="🎓")
    st.title("Crescent University Chatbot 🤖")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = {}
    if "dataset_loaded" not in st.session_state:
        embed_model, sym_spell, dataset, q_embeds = load_all_data()
        st.session_state.embed_model = embed_model
        st.session_state.sym_spell = sym_spell
        st.session_state.dataset = dataset
        st.session_state.q_embeds = q_embeds
        st.session_state.dataset_loaded = True

    # API key input
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""

    if not st.session_state.openai_api_key:
        with st.expander("Enter your OpenAI API Key (required for chatbot)"):
            key = st.text_input("OpenAI API Key", type="password")
            if key:
                st.session_state.openai_api_key = key.strip()
                openai.api_key = st.session_state.openai_api_key
    else:
        openai.api_key = st.session_state.openai_api_key

    # Chat display
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Bot:** {msg['content']}")

    # User input
    user_input = st.text_input("Ask me anything about Crescent University:", key="user_input")

    if user_input:
        if is_abusive(user_input):
            bot_reply = "Please avoid using offensive language."
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            st.experimental_rerun()

        # Normalize & expand
        norm_input = normalize_text(user_input, st.session_state.sym_spell)
        norm_input = resolve_follow_up(norm_input, st.session_state.memory)
        st.session_state.memory = update_chat_memory(norm_input, st.session_state.memory)

        # Handle greetings & farewells fast
        if is_greeting(user_input):
            bot_reply = get_random_greeting_response()
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            st.experimental_rerun()
        elif is_farewell(user_input):
            bot_reply = get_random_farewell_response()
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            st.experimental_rerun()

        # Generate response via GPT with retrieval + memory context
        memory_summary = summarize_memory(st.session_state.messages)
        knowledge_snippets = get_top_k_relevant_snippets(
            norm_input,
            st.session_state.dataset,
            st.session_state.embed_model,
            st.session_state.q_embeds,
            k=3
        )
        gpt_messages = build_gpt_prompt(norm_input, memory_summary, knowledge_snippets)

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=gpt_messages,
                temperature=0.5,
                max_tokens=400,
            )
            bot_reply = response.choices[0].message.content.strip()
        except AuthenticationError:
            bot_reply = "Invalid OpenAI API Key. Please check your key and try again."
        except Exception as e:
            bot_reply = f"Sorry, something went wrong: {str(e)}"

        # Append chat messages
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})

        st.experimental_rerun()

if __name__ == "__main__":
    main()
