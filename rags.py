import streamlit as st
import json
import re
import time
from symspellpy.symspellpy import SymSpell
import pkg_resources
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="Crescent Chatbot", layout="centered")

# Load your dataset (adjust path if needed)
with open("qa_dataset.json", "r") as f:
    data = json.load(f)

# Initialize SymSpell for spell correction
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

pidgin_dict_path = "pidgin_dict.txt"
if os.path.exists(pidgin_dict_path):
    sym_spell.load_dictionary(pidgin_dict_path, term_index=0, count_index=1)
else:
    logging.warning(f"Pidgin dictionary file {pidgin_dict_path} not found.")

abbreviations = {
    "u": "you", "r": "are", "ur": "your", "cn": "can", "cud": "could", "shud": "should", "wud": "would",
    "abt": "about", "bcz": "because", "plz": "please", "pls": "please", "tmrw": "tomorrow", "wat": "what",
    "wats": "what is", "info": "information", "yr": "year", "sem": "semester", "admsn": "admission",
    "clg": "college", "sch": "school", "uni": "university", "cresnt": "crescent", "l": "level", 
    "d": "the", "doc": "document", "msg": "message", "idk": "i don't know", "imo": "in my opinion",
    "asap": "as soon as possible", "dept": "department", "reg": "registration", "fee": "fees",
    "pg": "postgraduate", "app": "application", "req": "requirement", "nd": "national diploma",
    "a-level": "advanced level", "alevel": "advanced level", "2nd": "second", "1st": "first",
    "nxt": "next", "prev": "previous", "exp": "experience"
}

pidgin_map = {
    "how far": "how are you", "wetin": "what", "no wahala": "no problem", "abeg": "please",
    "sharp sharp": "quickly", "bros": "brother", "guy": "person", "waka": "walk", "chop": "eat",
    "jollof": "rice dish", "nah": "no", "dey": "is", "yarn": "talk", "gbam": "exactly", "ehn": "yes",
    "waka pass": "walk past", "how you dey": "how are you", "i no sabi": "i don't know",
    "make we go": "let's go", "omo": "child", "dash": "give", "carry go": "continue",
    "owo": "money", "pikin": "child", "see as e be": "look how it is", "no vex": "sorry",
    "sharp": "fast", "jare": "please", "e sure": "it is sure", "you sabi": "you know",
    "abeg make you": "please", "how you see am": "what do you think", "carry come": "bring"
}

abbreviations.update(pidgin_map)

synonym_map = {
    "lecturers": "academic staff", "professors": "academic staff", "teachers": "academic staff",
    "instructors": "academic staff", "tutors": "academic staff", "staff members": "staff",
    "head": "dean", "hod": "head of department", "dept": "department", "school": "university",
    "college": "faculty", "course": "subject", "class": "course", "subject": "course", 
    "unit": "credit", "credit unit": "unit", "course load": "unit", "non teaching": "non-academic",
    "nonteaching": "non-academic", "admin worker": "non-academic staff",
    "support staff": "non-academic staff", "clerk": "non-academic staff", 
    "receptionist": "non-academic staff", "secretary": "non-academic staff", 
    "office staff": "non-academic staff", "tech staff": "technical staff", 
    "it people": "technical staff", "lab helper": "technical staff", 
    "computer staff": "technical staff", "equipment handler": "technical staff", 
    "it guy": "technical staff", "hostel": "accommodation", "lodging": "accommodation", 
    "room": "accommodation", "school fees": "tuition", "acceptance fee": "admission fee",
    "fees": "tuition", "enrol": "apply", "join": "apply", "sign up": "apply", "admit": "apply",
    "requirement": "criteria", "conditions": "criteria", "needed": "required", 
    "needed for": "required for", "who handles": "who manages", 
    "who takes care of": "who manages", "computer sci": "computer science",
    "cs": "computer science", "eco": "economics", "stat": "statistics", 
    "phy": "physics", "bio": "biology", "chem": "chemistry", 
    "mass comm": "mass communication", "comm": "communication", "archi": "architecture",
    "exam": "examination", "tests": "assessments", "marks": "grades"
}

def normalize_text(text):
    text = text.lower()
    # Replace abbreviations & pidgin
    for abbr, full in abbreviations.items():
        text = re.sub(rf'\b{re.escape(abbr)}\b', full, text)
    # Replace synonyms
    for key, val in synonym_map.items():
        text = re.sub(rf'\b{re.escape(key)}\b', val, text)
    # Spell correction using SymSpell
    suggest = sym_spell.lookup_compound(text, max_edit_distance=2)
    return suggest[0].term if suggest else text

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
        font-size: 16px;">
        {message}
    </div><div style="clear: both;"></div>
    """

def generate_bot_response(user_message):
    # Normalize user input
    normalized = normalize_text(user_message)

    # TODO: Replace below with your real logic, e.g., embedding search + GPT call
    # For demo, we just echo normalized input with a canned response
    return f"Thanks for your question about '{normalized}'. We are processing your query."

def main():
    st.title(" Crescent University Chatbot")

    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "bot", "content": "Hello! Welcome to Crescent University chatbot. How can I help you today?"}
        ]
    if "typing" not in st.session_state:
        st.session_state.typing = False

    # Display all messages
    for msg in st.session_state.messages:
        is_user = (msg["role"] == "user")
        st.markdown(render_message(msg["content"], is_user=is_user), unsafe_allow_html=True)

    # Typing indicator
    if st.session_state.typing:
        st.markdown(render_message("<i>Bot is typing...</i>", is_user=False), unsafe_allow_html=True)

    # Text input box with key to clear later
    user_input = st.text_input("You:", key="user_text", placeholder="Ask me anything...")

    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Clear input box immediately
        st.session_state.user_text = ""

        # Show typing indicator
        st.session_state.typing = True
        st.experimental_rerun()

        # Simulate processing time or
