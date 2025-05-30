import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from openai import OpenAI
import re
from symspellpy.symspellpy import SymSpell
import pkg_resources
import time
import threading

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="Crescent Chatbot", layout="centered")

# Load Q&A dataset
with open("qa_dataset.json", "r") as f:
    data = json.load(f)

# Initialize SymSpell for spell correction
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Abbreviation and slang maps
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
synonym_map.update(synonym_map)

def normalize_text(text):
    text = text.lower()
    for abbr, full in abbreviations.items():
        text = re.sub(rf'\b{re.escape(abbr)}\b', full, text)
    for key, val in synonym_map.items():
        text = re.sub(rf'\b{re.escape(key)}\b', val, text)
    suggest = sym_spell.lookup_compound(text, max_edit_distance=2)
    return suggest[0].term if suggest else text

# Render chat bubbles with animation
def render_message(message, is_user=True):
    bg_color = "#DCF8C6" if is_user else "#E1E1E1"
    align = "right" if is_user else "left"
    margin = "10px 0 10px 50px" if is_user else "10px 50px 10px 0"
    animation_class = "slideInRight" if is_user else "slideInLeft"
    return f"""
    <div class="message {animation_class}" style="
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

# CSS for animations and typing indicator
st.markdown(
    """
    <style>
    .message {
        opacity: 0;
        animation-fill-mode: forwards;
        animation-duration: 0.5s;
        animation-timing-function: ease-out;
    }
    .slideInRight {
        animation-name: slideInRight;
    }
    .slideInLeft {
        animation-name: slideInLeft;
    }
    @keyframes slideInRight {
        from {
            transform: translateX(50%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    @keyframes slideInLeft {
        from {
            transform: translateX(-50%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    .typing-indicator {
        font-style: italic;
        color: gray;
        margin: 10px 0 10px 50px;
        clear: both;
        font-family: Arial, sans-serif;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "typing" not in st.session_state:
    st.session_state.typing = False
if "greeted" not in st.session_state:
    st.session_state.greeted = False

def chatbot_response(question):
    # Normalize input
    normalized_question = normalize_text(question)
    
    # Special handling: if user asks about courses at a level without semester, return both semesters
    level_match = re.search(r"(?:\b)(\d{3}|\d{2}|100|200|300|400|500|600|level|l)(?:\b)", normalized_question)
    if level_match:
        level = level_match.group(1)
        # Look for department
        dept_match = re.search(r"(computer science|cs|mass communication|comm|law|physics|chemistry|biology|architecture|economics|statistics|math|mathematics|engineering|history|english|education|accounting|management|business)", normalized_question)
        if dept_match:
            dept = dept_match.group(1)
            # Compose answer fetching all courses for level for this department (both semesters)
            # This assumes data format contains 'department', 'level', 'semester', 'course' fields
            courses = []
            for entry in data:
                # Normalize entry department and level for matching
                entry_dept = entry.get("department", "").lower()
                entry_level = str(entry.get("level", "")).lower()
                if dept in entry_dept and str(level) in entry_level:
                    courses.append(f"{entry.get('course_code','')} - {entry.get('course_title','')}")
            if courses:
                answer = f"Here are the courses for {dept.title()} {level} level (all semesters):\n" + "\n".join(courses)
                return answer
    
    # Else fallback: normal retrieval from data or GPT generation
    
    # Simple retrieval: find closest matching question in dataset (placeholder)
    # For now, just return a generic response or best matching from data
    
    # To keep this example short, we'll simulate with a dummy fallback
    return "Sorry, I am still learning. Could you please rephrase or ask something else?"

def show_typing():
    st.session_state.typing = True
    placeholder = st.empty()
    placeholder.markdown('<div class="typing-indicator">Bot is typing...</div>', unsafe_allow_html=True)
    time.sleep(1.5)  # simulate typing delay
    placeholder.empty()
    st.session_state.typing = False

def main():
    st.title("Crescent University Chatbot")
    
    # Sidebar clear chat button
    with st.sidebar:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.greeted = False
    
    # Greet user once
    if not st.session_state.greeted:
        user_greeting = "Hello!"
        bot_greeting = "Hi there! I am Crescent Chatbot. How can I help you today?"
        st.session_state.messages.append({"role": "user", "content": user_greeting})
        st.session_state.messages.append({"role": "bot", "content": bot_greeting})
        st.session_state.greeted = True
    
    # Display chat messages
    for msg in st.session_state.messages:
        is_user = msg["role"] == "user"
        st.markdown(render_message(msg["content"], is_user=is_user), unsafe_allow_html=True)
    
    # Input box
    user_input = st.text_input("You:", key="input", placeholder="Ask me anything...")
    
    if user_input:
        # Clear input immediately
        st.session_state.input = ""
        
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Show typing indicator (in a separate thread so UI can update)
        show_typing()
        
        # Get bot response
        response = chatbot_response(user_input)
        st.session_state.messages.append({"role": "bot", "content": response})
        
        # Rerun to display new messages (Streamlit automatically reruns on interaction)
        st.experimental_rerun()

if __name__ == "__main__":
    main()
