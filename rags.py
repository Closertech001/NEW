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

# Static list of departments for quick matching
DEPARTMENT_NAMES = [
    "Computer Science", "Mass Communication", "Law",
    "Microbiology", "Accounting", "Political Science",
    "Business Administration", "Business Admin"
]

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

# --------------------------
# TEXT NORMALIZATION
def normalize_text(text, sym_spell):
    text = text.lower()
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    corrected = suggestions[0].term if suggestions else text
    for abbr, full in ABBREVIATIONS.items():
        corrected = re.sub(rf'\b{re.escape(abbr)}\b', full, corrected)
    for syn, rep in SYNONYMS.items():
        corrected = re.sub(rf'\b{re.escape(syn)}\b', rep, corrected)
    return corrected

# --------------------------
# HELPER: GREETINGS / FAREWELLS
def is_greeting(text):
    return text.lower().strip() in [
        "hi", "hello", "hey", "good morning",
        "good afternoon", "good evening"
    ]

def get_random_greeting_response():
    return random.choice([
        "Hello! How can I assist you today?",
        "Hi there! What can I help you with?",
        "Hey! Feel free to ask me anything about Crescent University.",
        "Greetings! How may I be of service?",
        "Hello! Ready to help you with any questions."
    ])

def is_farewell(text):
    return text.lower().strip() in [
        "bye", "goodbye", "see you", "later", "farewell", "cya", "peace", "exit"
    ]

def get_random_farewell_response():
    return random.choice([
        "Goodbye! Have a great day!",
        "See you later! Feel free to come back anytime.",
        "Bye! Take care!",
        "Farewell! Let me know if you need anything else.",
        "Peace out! Hope to chat again soon."
    ])

# --------------------------
# MEMORY HELPERS
def update_chat_memory(norm_input: str, memory: dict) -> dict:
    """Extract department, level, topic from the normalized input."""
    # Department detection
    for dept in DEPARTMENT_NAMES:
        if re.search(rf"\b{re.escape(dept.lower())}\b", norm_input):
            memory["department"] = dept
            break

    # Generic "department of X" pattern
    dep_match = re.search(r"department of ([a-zA-Z &]+)", norm_input)
    if dep_match:
        memory["department"] = dep_match.group(1).title()

    # Level detection (100–500)
    lvl_match = re.search(r"(100|200|300|400|500)\s*level", norm_input)
    if lvl_match:
        memory["level"] = lvl_match.group(1)

    # Topic detection via keywords
    TOPIC_KWS = {
        "admission": ["admission", "apply", "jamb", "requirement"],
        "fees": ["fee", "tuition", "cost", "school fees"],
        "courses": ["course", "subject", "unit", "curriculum", "study"],
        "accommodation": ["accommodation", "hostel", "reside", "lodging"],
        "graduation": ["graduation", "convocation"],
        "exam": ["exam", "test", "cgpa", "grade"],
        "scholarship": ["scholarship", "aid", "bursary"],
        "dress code": ["dress code", "uniform", "appearance"]
    }
    for topic, kws in TOPIC_KWS.items():
        if any(kw in norm_input for kw in kws):
            memory["topic"] = topic
            break

    return memory

def resolve_follow_up(raw_input: str, memory: dict) -> str:
    """
    Rewrite vague follow-up questions using the short-term memory,
    otherwise return the original input unchanged.
    """
    text = raw_input.strip().lower()

    # Example: "what about 300 level?"
    m = re.match(r"what about (\d{3}) level", text)
    if m and memory.get("department"):
        level = m.group(1)
        return f"What are the {level} level courses in {memory['department']}?"

    # Example: "what about 300 level" with level already in memory
    if text.startswith("what about") and memory.get("level") and memory.get("department"):
        return f"What are the {memory['level']} level courses in {memory['department']}?"

    # Example: "do they also do that in law?"
    m2 = re.match(r"do they also .* in ([a-zA-Z &]+)\?", text)
    if m2 and memory.get("topic"):
        dept = m2.group(1).title()
        return f"Do they also offer {memory['topic']} in {dept}?"

    # Example: "how about the same thing for Mass Comm?"
    m3 = re.match(r"how about .* for ([a-zA-Z &]+)\?", text)
    if m3 and memory.get("topic"):
        dept = m3.group(1).title()
        return f"Do they also offer {memory['topic']} in {dept}?"

    return raw_input  # unchanged

# --------------------------
# RAG RETRIEVAL
def retrieve_answer(user_input, dataset, q_embeds, embed_model):
    """Return best-matched answer & score from dataset using embeddings."""
    for item in dataset:
        if user_input.strip().lower() in item["question"].strip().lower():
            return item["answer"], 1.0

    user_embed = embed_model.encode(
        user_input, convert_to_tensor=True, normalize_embeddings=True
    )
    scores = util.pytorch_cos_sim(user_embed, q_embeds)[0]
    best_idx = scores.argmax().item()  # FIXED: extract scalar with .item()
    best_score = float(scores[best_idx])
    return dataset[best_idx]["answer"], best_score

# --------------------------
# GPT PROMPT
def build_contextual_prompt(messages, memory, max_turns=6):
    recent = messages[-max_turns * 2 :] if len(messages) > max_turns * 2 else messages
    mem_info = (
        f"Conversation context:\n"
        f"- Department: {memory.get('department') or 'unspecified'}\n"
        f"- Level: {memory.get('level') or 'unspecified'}\n"
        f"- Topic: {memory.get('topic') or 'unspecified'}"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant for Crescent University.\n"
                + mem_info
            ),
        }
    ] + recent

# --------------------------
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

    # Load models/data once
    if "embed_model" not in st.session_state:
        embed_model, sym_spell, dataset, q_embeds = load_all_data()
        st.session_state.embed_model = embed_model
        st.session_state.sym_spell = sym_spell
        st.session_state.dataset = dataset
        st.session_state.q_embeds = q_embeds
        st.session_state.openai_enabled = openai_enabled

    # Conversation & memory states
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I'm your Crescent University assistant. "
                    "Ask me anything!"
                ),
            }
        ]
    if "short_term_memory" not in st.session_state:
        st.session_state.short_term_memory = {
            "department": None,
            "topic": None,
            "level": None,
        }

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Get new user input
    user_input = st.chat_input("Type your question here...")

    if user_input:
        # 1️⃣ Normalize & profanity check
        norm_input = normalize_text(user_input, st.session_state.sym_spell)

        if ABUSE_PATTERN.search(norm_input):
            response = (
                "Sorry, I can’t help with that. "
                "Try asking about something academic."
            )
        elif is_greeting(norm_input):
            response = get_random_greeting_response()
        elif is_farewell(norm_input):
            response = get_random_farewell_response()
        else:
            # 2️⃣ Update memory from the new input
            st.session_state.short_term_memory = update_chat_memory(
                norm_input, st.session_state.short_term_memory
            )

            # 3️⃣ Resolve vague follow-ups using memory
            search_input = resolve_follow_up(
                user_input, st.session_state.short_term_memory
            )

            # 4️⃣ Retrieval-augmented search
            answer, score = retrieve_answer(
                search_input,
                st.session_state.dataset,
                st.session_state.q_embeds,
                st.session_state.embed_model,
            )

            # Save user message to history before GPT
            st.session_state.messages.append({"role": "user", "content": user_input})

            # 5️⃣ Decide: direct answer or GPT fallback
            if score >= 0.50:
                response = answer
            elif score >= 0.45 and st.session_state.openai_enabled:
                gpt_prompt = build_contextual_prompt(
                    st.session_state.messages, st.session_state.short_term_memory
                )
                try:
                    gpt_response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=gpt_prompt,
                        temperature=0.3,
                    )
                    response = gpt_response.choices[0].message.content
                except AuthenticationError:
                    response = (
                        "Sorry, GPT is not available due to a configuration issue."
                    )
                except Exception:
                    response = (
                        "Sorry, I encountered an issue while trying to answer that."
                    )
            else:
                response = (
                    "Hmm, I’m not sure what you mean. "
                    "Can you rephrase or ask differently?"
                )

        # 6️⃣ Display user + assistant messages
        st.chat_message("user").markdown(user_input)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("_Bot is typing..._")
            time.sleep(1.2)
            placeholder.markdown(response)

        # 7️⃣ Add assistant reply to history
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Sidebar debug (optional)
        # st.sidebar.write("🔁 Short-Term Memory")
        # st.sidebar.json(st.session_state.short_term_memory)

if __name__ == "__main__":
    main()
