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
from openai._exceptions import AuthenticationError
import openai
from textblob import TextBlob

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

openai.api_key = os.getenv("OPENAI_API_KEY")

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

# --- Helpers ---

def normalize_text(text, sym_spell):
    text = text.lower()
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    corrected = suggestions[0].term if suggestions else text
    for abbr, full in ABBREVIATIONS.items():
        corrected = re.sub(rf'{re.escape(abbr)}', full, corrected)
    for syn, rep in SYNONYMS.items():
        corrected = re.sub(rf'{re.escape(syn)}', rep, corrected)
    return corrected

def is_greeting(text):
    return any(kw in text for kw in ["hi", "hello", "hey", "good morning", "greetings"])

def get_random_greeting_response():
    return random.choice([
        "Hi there! How can I help you today?",
        "Hello! Ask me anything about Crescent University.",
        "Hey! What would you like to know?"
    ])

def is_farewell(text):
    return any(kw in text for kw in ["bye", "goodbye", "see you", "take care"])

def get_random_farewell_response():
    return random.choice([
        "Goodbye! Let me know if you need anything else.",
        "Take care! I'm always here to help.",
        "Bye! Have a great day ahead."
    ])

def enrich_response(response, memory):
    suggestions = [
        "Would you like to know about the fees or accommodation?",
        "Should I help you with admission requirements too?",
        "Let me know if you want details on the exam process or CGPA."
    ]
    if any(k in response.lower() for k in ["course", "unit", "siwes"]):
        return response + "\n\n" + random.choice(suggestions)
    return response

# --- Memory & Clarification ---
def update_chat_memory(norm_input, memory):
    for dept in DEPARTMENT_NAMES:
        if re.search(rf"\b{re.escape(dept)}\b", norm_input):
            memory["department"] = dept.title()
    if m := re.search(r"department of ([a-zA-Z &]+)", norm_input):
        memory["department"] = m.group(1).title()
    if m := re.search(r"(100|200|300|400|500)\s*level", norm_input):
        memory["level"] = m.group(1)
    topics = [
        ("admission", ["admission", "apply", "jamb", "requirement"]),
        ("fees", ["fee", "tuition", "cost", "school fees"]),
        ("courses", ["course", "subject", "unit", "curriculum", "study"]),
        ("accommodation", ["accommodation", "hostel", "reside", "lodging"]),
        ("graduation", ["graduation", "convocation"]),
        ("exam", ["exam", "test", "cgpa", "grade"]),
        ("scholarship", ["scholarship", "aid", "bursary"]),
        ("siwes", ["siwes", "internship", "industrial training"]),
        ("dress code", ["dress code", "uniform", "appearance"])
    ]
    for topic, kws in topics:
        if any(kw in norm_input for kw in kws):
            memory["topic"] = topic
    memory["last_question"] = norm_input
    return memory

def resolve_follow_up(raw_input, memory):
    text = raw_input.strip().lower()
    if m := re.match(r"(how|what) about (\d{3}) level", text):
        level = m.group(2)
        dept = memory.get("department")
        topic = memory.get("topic")
        return f"What about {topic} in {dept} for {level} level?" if topic and dept else raw_input
    return raw_input

def build_contextual_prompt(messages, memory):
    context = f"- Department: {memory.get('department') or 'unspecified'}\n- Level: {memory.get('level') or 'unspecified'}\n- Topic: {memory.get('topic') or 'unspecified'}"
    prompt = [{"role": "system", "content": "You are a helpful assistant for Crescent University.\n" + context}]
    for pair in memory.get("conversation_history", []):
        prompt.append({"role": "user", "content": pair["user"]})
        prompt.append({"role": "assistant", "content": pair["bot"]})
    return prompt[-12:]

def paraphrase_with_gpt(answer, user_input):
    try:
        messages = [
            {"role": "system", "content": "You are a friendly assistant. Rephrase the answer below in a warm, natural tone suitable for chatting with a student. Make it clear and concise."},
            {"role": "user", "content": f"Question: {user_input}\nAnswer: {answer}"}
        ]
        gpt_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7
        )
        return gpt_response.choices[0].message.content.strip()
    except Exception as e:
        print("Paraphrasing error:", e)
        return answer

def retrieve_or_gpt(user_input, dataset, q_embeds, embed_model, messages, memory):
    for item in dataset:
        if user_input.strip().lower() == item["question"].strip().lower():
            return item["answer"], 1.0
    user_embed = embed_model.encode(user_input, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.pytorch_cos_sim(user_embed, q_embeds)[0]
    top_score = float(scores.max())
    best_idx = scores.argmax().item()
    if top_score >= 0.60:
        answer = dataset[best_idx]["answer"]
        if openai.api_key:
            answer = paraphrase_with_gpt(answer, user_input)
        return answer, top_score
    elif top_score < 0.4:
        clarification = suggest_clarification(user_input.lower(), memory)
        return clarification, top_score
    if openai.api_key:
        try:
            prompt = build_contextual_prompt(messages, memory)
            prompt.append({"role": "user", "content": user_input})
            gpt_response = openai.ChatCompletion.create(model="gpt-4", messages=prompt, temperature=0.4)
            return gpt_response.choices[0].message.content, top_score
        except Exception as e:
            print(f"GPT error: {e}")
            return "Sorry, I couldn’t fetch a proper response right now.", top_score
    return "I'm not sure what you mean. Could you try rephrasing?", top_score

def log_to_long_term_memory(user_input, assistant_response):
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/chat_history_log.json"
    memory_path = "logs/long_term_memory.json"
    entry = {"user": user_input, "assistant": assistant_response, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        if os.path.exists(log_path):
            with open(log_path, "r+") as f:
                data = json.load(f)
                data.append(entry)
                f.seek(0)
                json.dump(data, f, indent=2)
        else:
            with open(log_path, "w") as f:
                json.dump([entry], f, indent=2)
        memory = {}
        if os.path.exists(memory_path):
            with open(memory_path, "r") as f:
                memory = json.load(f)
        user_id = "default_user"
        if user_id not in memory:
            memory[user_id] = {}
        mem = memory[user_id]
        for key in ["department", "topic", "level"]:
            if st.session_state.memory.get(key):
                mem[key] = st.session_state.memory[key]
        with open(memory_path, "w") as f:
            json.dump(memory, f, indent=2)
    except Exception as e:
        print("Logging error:", e)

def main():
    st.set_page_config(page_title="Crescent University Chatbot", page_icon="🎓")
    st.title(":mortar_board: Crescent University Chatbot")
    st.markdown("Ask me anything about your department, courses, or university life.")

    if st.button("Reset Chat"):
        for key in ["messages", "memory"]:
            st.session_state.pop(key, None)
        st.experimental_rerun()

    if "embed_model" not in st.session_state:
        embed_model = load_embedding_model()
        sym_spell = init_symspell()
        dataset = load_dataset()
        q_embeds = embed_model.encode([item["question"] for item in dataset], convert_to_tensor=True, normalize_embeddings=True)
        st.session_state.embed_model = embed_model
        st.session_state.sym_spell = sym_spell
        st.session_state.dataset = dataset
        st.session_state.q_embeds = q_embeds
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your Crescent University assistant. Ask me anything!"}]
        st.session_state.memory = {"department": None, "topic": None, "level": None, "conversation_history": []}
        try:
            memory_path = "logs/long_term_memory.json"
            if os.path.exists(memory_path):
                with open(memory_path, "r") as f:
                    memory_data = json.load(f)
                    user_id = "default_user"
                    if user_id in memory_data:
                        st.session_state.memory.update(memory_data[user_id])
        except:
            pass

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Type your question here...")
    if user_input:
        norm_input = normalize_text(user_input, st.session_state.sym_spell)
        if ABUSE_PATTERN.search(norm_input):
            response = "Sorry, I can’t help with that."
        elif is_greeting(norm_input):
            response = get_random_greeting_response()
        elif is_farewell(norm_input):
            response = get_random_farewell_response()
        elif is_small_talk(norm_input):
            response = random.choice([
                "I'm doing great! Just here to help you with Crescent University questions. 😊",
                "You're very welcome! I'm happy to assist you anytime.",
                "I was created by a developer using OpenAI tools to assist Crescent University students.",
                "Sure! Did you know that Crescent University is known for its serene academic environment?",
                "I can help you with admissions, courses, fees, accommodation, and lots more.",
                "Thank you! That means a lot. I'm always improving to serve you better. 🙌"
            ])
        else:
            st.session_state.memory = update_chat_memory(norm_input, st.session_state.memory)
            resolved_input = resolve_follow_up(user_input, st.session_state.memory)
            response, _ = retrieve_or_gpt(resolved_input, st.session_state.dataset, st.session_state.q_embeds, st.session_state.embed_model, st.session_state.messages, st.session_state.memory)
            response = enrich_response(response, st.session_state.memory)

        st.chat_message("user").markdown(user_input)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("🤖 _Typing..._")
            delay = min(2.5, max(0.8, len(response) / 80))
            time.sleep(delay)
            placeholder.markdown(response)

        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.memory["conversation_history"].append({"user": user_input, "bot": response})
        st.session_state.memory["conversation_history"] = st.session_state.memory["conversation_history"][-5:]
        log_to_long_term_memory(user_input, response)

if __name__ == "__main__":
    main()
