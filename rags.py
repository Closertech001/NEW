import re
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from openai import OpenAI
import logging
import pkg_resources
from symspellpy.symspellpy import SymSpell

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Set Streamlit page config first
st.set_page_config(page_title="Crescent Chatbot", layout="centered")

# Load structured dataset
with open("qa_dataset.json", "r") as f:
    data = json.load(f)

# Use entire dataset
filtered_data = data

# Initialize SymSpell for spell correction (and pidgin)
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

pidgin_dict_path = "pidgin_dict.txt"
try:
    sym_spell.load_dictionary(pidgin_dict_path, term_index=0, count_index=1)
except Exception:
    logging.warning(f"Pidgin dictionary file {pidgin_dict_path} not found. Skipping pidgin spell corrections.")

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

# Add common Pidgin English phrases and slang
pidgin_map = {
    "how far": "how are you",
    "wetin": "what",
    "no wahala": "no problem",
    "abeg": "please",
    "sharp sharp": "quickly",
    "bros": "brother",
    "guy": "person",
    "waka": "walk",
    "chop": "eat",
    "jollof": "rice dish",
    "nah": "no",
    "dey": "is",
    "yarn": "talk",
    "gbam": "exactly",
    "ehn": "yes",
    "waka pass": "walk past",
    "how you dey": "how are you",
    "i no sabi": "i don't know",
    "make we go": "let's go",
    "omo": "child",
    "dash": "give",
    "carry go": "continue",
    "owo": "money",
    "pikin": "child",
    "see as e be": "look how it is",
    "no vex": "sorry",
    "sharp": "fast",
    "jare": "please",
    "e sure": "it is sure",
    "you sabi": "you know",
    "abeg make you": "please",
    "how you see am": "what do you think",
    "carry come": "bring",
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
    for abbr, full in abbreviations.items():
        text = re.sub(rf'\b{re.escape(abbr)}\b', full, text)
    for key, val in synonym_map.items():
        text = re.sub(rf'\b{re.escape(key)}\b', val, text)
    suggest = sym_spell.lookup_compound(text, max_edit_distance=2)
    return suggest[0].term if suggest else text

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
model = load_model()

@st.cache_resource
def build_index():
    questions = [normalize_text(qa["question"]) for qa in filtered_data]
    emb = model.encode(questions, show_progress_bar=False)
    emb = np.array(emb).astype("float32")
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(emb.shape[1]), emb.shape[1], 100)
    index.train(emb)
    index.add(emb)
    return index, emb, questions

index, embeddings, questions = build_index()

# Helper extractors

def extract_level(text):
    match = re.search(r'\b(\d{3})\s*level\b', text)
    if match:
        return match.group(1)
    return None

def extract_semester(text):
    sem_map = {
        'first': 'first',
        '1st': 'first',
        'second': 'second',
        '2nd': 'second',
        '1': 'first',
        '2': 'second',
    }
    for key in sem_map:
        if key in text:
            return sem_map[key]
    return None

def extract_department(text):
    # List your departments here — make sure to lowercase for matching
    departments = [
        'computer science', 'mass communication', 'law', 'physics', 'chemistry',
        'biology', 'architecture', 'economics', 'statistics', 'mathematics',
        'engineering', 'accounting', 'education'
    ]
    for dept in departments:
        if dept in text:
            return dept
    return None

def get_courses_by_level_and_dept(level, department, semester=None):
    results = []
    for entry in data:
        entry_level = str(entry.get("level", "")).lower()
        entry_semester = str(entry.get("semester", "")).lower()
        entry_dept = str(entry.get("department", "")).lower()

        if entry_level == level.lower() and department.lower() in entry_dept:
            if semester:
                if semester.lower() == entry_semester:
                    results.append(entry)
            else:
                results.append(entry)

    if not results:
        return f"Sorry, I couldn't find any courses for {level} level in {department}."

    # Format the course list nicely
    course_lines = []
    for c in results:
        sem_text = c.get("semester", "N/A").capitalize()
        code = c.get("course_code", "N/A").upper()
        name = c.get("course_name", "Unnamed Course")
        course_lines.append(f"- {code}: {name} ({sem_text} semester)")

    return f"Courses for {level} level in {department}:\n" + "\n".join(course_lines)

def render_message(message, is_user=True):
    bg_color = "#DCF8C6" if is_user else "#E1E1E1"
    align = "right" if is_user else "left"
    margin = "10px 0 10px 50px" if is_user else "10px 50px 10px 0"
    return f"""
    <style>
    @keyframes fadeIn {{
        0% {{ opacity: 0; transform: translateY(10px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
    }}
    .chat-message {{
        animation: fadeIn 0.4s ease-in-out;
    }}
    </style>

    <div class="chat-message" style="
        background-color: {bg_color};
        padding: 10px;
        border-radius: 10px;
        max-width: 70%;
        margin: {margin};
        text-align: left;
        float: {align};
        clear: both;
        font-family: Arial, sans-serif;
        font-size: 16px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">
        {message}
    </div><div style="clear: both;"></div>
    """

# Main app function

def main():
    st.title("Crescent University Chatbot")

    if "history" not in st.session_state:
        st.session_state.history = []

    # Sidebar clear chat button
    if st.sidebar.button("Clear Chat"):
        st.session_state.history = []
        st.experimental_rerun()

    user_input = st.text_input("Ask me anything about Crescent University:")

    if user_input:
        norm_input = normalize_text(user_input)

        level = extract_level(norm_input)
        department = extract_department(norm_input)
        semester = extract_semester(norm_input)

        if level and department:
            answer = get_courses_by_level_and_dept(level, department, semester)
        else:
            # Your existing logic for course code or fallback search
            course_code = None
            match_course = re.search(r'\b([A-Za-z]{3}\s?\d{3})\b', norm_input)
            if match_course:
                course_code = match_course.group(1).replace(" ", "").upper()

            if course_code:
                # Your get_course_info function or fallback logic here
                answer = get_course_info(course_code)
            else:
                # Search embedding index
                query_emb = model.encode([norm_input], show_progress_bar=False)
                D, I = index.search(np.array(query_emb).astype("float32"), 10)
                best_score = D[0][0]
                best_idx = I[0][0]

                if best_score < 1.0 and best_idx < len(filtered_data):
                    answer = filtered_data[best_idx]["answer"]
                else:
                    answer = rag_fallback_with_context(user_input, I[0])
                    if "couldn't find" in answer.lower() or "try rephrasing" in answer.lower():
                        log_unmatched_query(user_input)

        st.session_state.history.append(("user", user_input))
        st.session_state.history.append(("bot", answer))

    for role, msg in st.session_state.history:
        st.markdown(render_message(msg, is_user=(role == "user")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
