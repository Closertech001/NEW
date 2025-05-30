import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from openai import OpenAI
import re
from symspellpy.symspellpy import SymSpell
import pkg_resources

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

def normalize_text(text):
    text = text.lower()
    # Replace abbreviations and slang word-by-word
    words = text.split()
    normalized_words = []
    for w in words:
        w = abbreviations.get(w, w)
        w = synonym_map.get(w, w)
        normalized_words.append(w)
    normalized_text = " ".join(normalized_words)
    # Use symspell to correct spelling
    suggest = sym_spell.lookup_compound(normalized_text, max_edit_distance=2)
    return suggest[0].term if suggest else normalized_text

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
    normalized_question = normalize_text(question)

    # Try to detect level (only digits like 100, 200, ... or 2-digit numbers)
    level_match = re.search(r"\b(100|200|300|400|500|600|\d{2,3})\b", normalized_question)
    level = level_match.group(1) if level_match else None

    # Try to detect department from known list
    departments_list = [
        "computer science", "cs", "mass communication", "comm", "law",
        "physics", "chemistry", "biology", "architecture", "economics",
        "statistics", "math", "mathematics", "engineering", "history",
        "english", "education", "accounting", "management", "business"
    ]
    dept = None
    for d in departments_list:
        if d in normalized_question:
            dept = d
            break

    if level and dept:
        courses = []
        # Normalize department name for matching in data keys (lowercase)
        dept_norm = dept.lower()
        level_norm = str(level).lower()

        for entry in data:
            entry_dept = entry.get("department", "").lower()
            entry_level = str(entry.get("level", "")).lower()

            # Match exact level and dept
            if dept_norm == entry_dept and level_norm == entry_level:
                course_code = entry.get("course_code", "").strip()
                course_title = entry.get("course_title", "").strip()
                if course_code and course_title:
                    courses.append(f"{course_code} - {course_title}")

        if courses:
            answer = f"Here are the courses for {dept.title()} {level} level (all semesters):\n" + "\n".join(courses)
            return answer

    # Fallback generic response
    return "Sorry, I couldn't find courses for that department and level. Could you please clarify?"

def main():
    st.title("Crescent University Chatbot")

    with st.sidebar:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.greeted = False
            st.session_state.typing = False
            st.experimental_rerun()

    if not st.session_state.greeted:
        greeting_msg = "Hi there! I am Crescent Chatbot. How can I help you today?"
        st.session_state.messages.append({"role": "bot", "content": greeting_msg})
        st.session_state.greeted = True

    # Display chat messages
    for msg in st.session_state.messages:
        is_user = msg["role"] == "user"
        st.markdown(render_message(msg["content"], is_user), unsafe_allow_html=True)

    # Show typing indicator if bot is typing
    if st.session_state.typing:
        st.markdown('<div class="typing-indicator">Bot is typing...</div>', unsafe_allow_html=True)

    # User input box
    user_input = st.text_input("You:", key="input", placeholder="Ask me anything...")

    # When user inputs text and presses Enter
    if user_input and user_input.strip():
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input.strip()})
        # Clear input box immediately
        st.session_state.input = ""
        # Show typing indicator
        st.session_state.typing = True
        st.experimental_rerun()  # rerun to show typing indicator

    # After showing typing indicator, generate bot response and update messages
    if st.session_state.typing and st.session_state.messages:
        last_msg = st.session_state.messages[-1]
        if last_msg["role"] == "user":
            response = chatbot_response(last_msg["content"])
            st.session_state.messages.append({"role": "bot", "content": response})
            st.session_state.typing = False
            st.experimental_rerun()

if __name__ == "__main__":
    main()
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

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="Crescent Chatbot", layout="centered")

# Load Q&A dataset
with open("qa_dataset.json", "r") as f:
    data = json.load(f)

# Prepare data: extract questions and answers
questions = [entry["question"] for entry in data]
answers = [entry["answer"] for entry in data]

# Load Sentence Transformer model for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Compute embeddings for all dataset questions
question_embeddings = embedder.encode(questions, convert_to_numpy=True)

# Build FAISS index for fast similarity search
dimension = question_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity if embeddings normalized)
faiss.normalize_L2(question_embeddings)
index.add(question_embeddings)

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

def openai_fallback_response(question):
    prompt = f"""
You are Crescent University’s friendly and knowledgeable chatbot assistant. Your role is to help students, staff, and visitors by answering questions about courses, admissions, departments, fees, registration, campus services, and general university info. Always respond politely and clearly. If you don’t know the answer, say:

"I’m sorry, I don’t have that information right now. Please check the university’s official website or contact the administration for further assistance."

Feel free to ask the user if they need clarification or more help.

User question: "{question}"

Examples:

Q: What courses are offered in Computer Science 100 level?  
A: The courses include Introduction to Programming, Mathematics for Computing, and Computer Science Fundamentals.

Q: How can I apply for admission?  
A: You can apply for admission through the university’s online portal at admissions.crescentuniversity.edu.

Answer:
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful university assistant chatbot."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300,
        n=1,
    )
    return response.choices[0].message.content.strip()

# Initialize session state for chat history and input box clearing
if "messages" not in st.session_state:
    st.session_state.messages = []
if "input" not in st.session_state:
    st.session_state.input = ""

# Greeting on first load
if len(st.session_state.messages) == 0:
    greeting = "Hello! I’m Crescent University Chatbot. How can I assist you today?"
    st.session_state.messages.append({"content": greeting, "is_user": False})

st.title("Crescent University Chatbot")

def get_answer(user_question):
    normalized = normalize_text(user_question)
    query_emb = embedder.encode([normalized], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)
    D, I = index.search(query_emb, k=3)  # Top 3 matches
    top_indices = I[0]
    top_scores = D[0]

    # Threshold for similarity (cosine similarity)
    threshold = 0.65
    for idx, score in zip(top_indices, top_scores):
        if score >= threshold:
            return answers[idx]
    # If no good match, fallback to GPT
    return openai_fallback_response(user_question)

def main():
    # Display chat history
    for msg in st.session_state.messages:
        st.markdown(render_message(msg["content"], msg["is_user"]), unsafe_allow_html=True)

    # Input form with cleared input after send
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your message here...", value="", key="input_box")
        submit = st.form_submit_button("Send")

        if submit and user_input.strip():
            st.session_state.messages.append({"content": user_input.strip(), "is_user": True})
            with st.spinner("Bot is typing..."):
                bot_response = get_answer(user_input.strip())
                time.sleep(0.8)  # Small delay for typing feel
            st.session_state.messages.append({"content": bot_response, "is_user": False})
            st.experimental_rerun()

if __name__ == "__main__":
    main()
