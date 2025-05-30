import streamlit as st
import re
import time
import random
import json
from sentence_transformers import SentenceTransformer, util
from symspellpy.symspellpy import SymSpell
import pkg_resources

# --------------------------
# Load data
@st.cache_resource
def load_dataset():
    with open("QA_dataset.json", "r") as f:
        return json.load(f)

# --------------------------
# Initialize SymSpell for typo correction
@st.cache_resource
def init_symspell():
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dict_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
    return sym_spell

# --------------------------
# Load SentenceTransformer embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------
# Normalization dictionaries for abbreviations, slang, synonyms
ABBREVIATIONS = {
    "u": "you", "r": "are", "ur": "your", "cn": "can", "cud": "could", "shud": "should", "wud": "would",
    "abt": "about", "bcz": "because", "plz": "please", "pls": "please", "tmrw": "tomorrow", "wat": "what",
    "wats": "what is", "info": "information", "yr": "year", "sem": "semester", "admsn": "admission",
    "clg": "college", "sch": "school", "uni": "university", "cresnt": "crescent", "l": "level", 
    "d": "the", "msg": "message", "idk": "i don't know", "imo": "in my opinion", "asap": "as soon as possible",
    "dept": "department", "reg": "registration", "fee": "fees", "pg": "postgraduate", "app": "application",
    "req": "requirement", "nd": "national diploma", "a-level": "advanced level", "alevel": "advanced level",
    "2nd": "second", "1st": "first", "nxt": "next", "prev": "previous", "exp": "experience",
    # Pidgin & informal
    "how far": "how are you", "wetin": "what", "no wahala": "no problem", "abeg": "please",
    "sharp sharp": "quickly", "bros": "brother", "guy": "person", "waka": "walk", "chop": "eat",
    "jollof": "rice dish", "nah": "no", "dey": "is", "yarn": "talk", "gbam": "exactly", "ehn": "yes",
    "waka pass": "walk past", "how you dey": "how are you", "i no sabi": "i don't know",
    "make we go": "let's go", "omo": "child", "dash": "give", "carry go": "continue", "owo": "money",
    "pikin": "child", "see as e be": "look how it is", "no vex": "sorry", "sharp": "fast", 
    "jare": "please", "e sure": "it is sure", "you sabi": "you know", "abeg make you": "please", 
    "how you see am": "what do you think", "carry come": "bring"
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
    "needed": "required", "needed for": "required for", "who handles": "who manages", 
    "cs": "computer science", "eco": "economics", "stat": "statistics", "phy": "physics",
    "bio": "biology", "chem": "chemistry", "mass comm": "mass communication", 
    "archi": "architecture", "exam": "examination", "marks": "grades"
}

# --------------------------
# Normalize input text
def normalize_text(text, sym_spell):
    text = text.lower()
    # Replace abbreviations
    for abbr, full in ABBREVIATIONS.items():
        text = re.sub(rf'\b{re.escape(abbr)}\b', full, text)
    # Replace synonyms
    for syn, rep in SYNONYMS.items():
        text = re.sub(rf'\b{re.escape(syn)}\b', rep, text)
    # Spell correction using symspell compound lookup
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    if suggestions:
        return suggestions[0].term
    else:
        return text

# --------------------------
# Greeting & Farewell detection
def is_greeting(text):
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    return any(text.lower().strip() == g for g in greetings)

def get_random_greeting_response():
    return random.choice([
        "Hello! How can I assist you today?",
        "Hi there! What can I help you with?",
        "Hey! Feel free to ask me anything about Crescent University.",
        "Greetings! How may I be of service?",
        "Hello! Ready to help you with any questions."
    ])

def is_farewell(text):
    farewells = ["bye", "goodbye", "see you", "later", "farewell", "cya", "peace", "exit"]
    return any(text.lower().strip() == f for f in farewells)

def get_random_farewell_response():
    return random.choice([
        "Goodbye! Have a great day!",
        "See you later! Feel free to come back anytime.",
        "Bye! Take care!",
        "Farewell! Let me know if you need anything else.",
        "Peace out! Hope to chat again soon."
    ])

# --------------------------
# Retrieve best matching Q&A
def retrieve_answer(user_input, dataset, embed_model, top_k=1):
    user_embed = embed_model.encode(user_input, convert_to_tensor=True)
    questions = [item["question"] for item in dataset]
    q_embeds = embed_model.encode(questions, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(user_embed, q_embeds)[0]
    best_idx = int(scores.argmax())
    return dataset[best_idx]["question"], dataset[best_idx]["answer"]

# --------------------------
# Main app
def main():
    st.set_page_config(page_title="Crescent University Chatbot", page_icon="🎓")
    st.title("🎓 Crescent University Chatbot")
    st.markdown("Ask me anything about your department, courses, or the university.")

    dataset = load_dataset()
    embed_model = load_embedding_model()
    sym_spell = init_symspell()

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your Crescent University assistant. Ask me anything!"}]

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Type your question here...")

    if user_input:
        norm_input = normalize_text(user_input, sym_spell)

        # Handle greetings
        if is_greeting(norm_input):
            response = get_random_greeting_response()
        # Handle farewells
        elif is_farewell(norm_input):
            response = get_random_farewell_response()
        else:
            # Retrieve best matching Q&A from dataset
            matched_q, answer = retrieve_answer(norm_input, dataset, embed_model)
            response = f"**Q:** {matched_q}\n\n**A:** {answer}"

        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Show user message
        st.chat_message("user").markdown(user_input)

        # Bot typing animation
        with st.chat_message("assistant"):
            typing_placeholder = st.empty()
            typing_placeholder.markdown("_Bot is typing..._")
            time.sleep(1.5)
            typing_placeholder.markdown(response)
        # Append bot message
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
