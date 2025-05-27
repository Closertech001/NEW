import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import random
import re
from symspellpy.symspellpy import SymSpell, Verbosity
import pkg_resources
import json
import openai
import os
import uuid

# --- API Key Setup ---
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- SymSpell Setup ---
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# --- Abbreviations and Department Mapping ---
abbreviations = {
    "u": "you", "r": "are", "ur": "your", "ow": "how", "pls": "please", "plz": "please",
    "tmrw": "tomorrow", "cn": "can", "wat": "what", "cud": "could", "shud": "should",
    "wud": "would", "abt": "about", "bcz": "because", "btw": "between", "asap": "as soon as possible",
    "idk": "i don't know", "imo": "in my opinion", "msg": "message", "doc": "document", "d": "the",
    "yr": "year", "sem": "semester", "dept": "department", "admsn": "admission",
    "cresnt": "crescent", "uni": "university", "clg": "college", "sch": "school",
    "info": "information", "l": "level", "CSC": "Computer Science", "ECO": "Economics with Operations Research",
    "PHY": "Physics", "STAT": "Statistics", "1st": "First", "2nd": "Second"
}

department_map = {
    "GST": "General Studies", "MTH": "Mathematics", "PHY": "Physics", "STA": "Statistics",
    "COS": "Computer Science", "CUAB-CSC": "Computer Science", "CSC": "Computer Science",
    "IFT": "Computer Science", "SEN": "Software Engineering", "ENT": "Entrepreneurship",
    "CYB": "Cybersecurity", "ICT": "Information and Communication Technology",
    "DTS": "Data Science", "CUAB-CPS": "Computer Science", "CUAB-ECO": "Economics with Operations Research",
    "ECO": "Economics with Operations Research", "SSC": "Social Sciences", "CUAB-BCO": "Economics with Operations Research",
    "LIB": "Library Studies", "LAW": "Law (BACOLAW)", "GNS": "General Studies", "ENG": "English",
    "SOS": "Sociology", "PIS": "Political Science", "CPS": "Computer Science",
    "LPI": "Law (BACOLAW)", "ICL": "Law (BACOLAW)", "LPB": "Law (BACOLAW)", "TPT": "Law (BACOLAW)",
    "FAC": "Agricultural Sciences", "ANA": "Anatomy", "BIO": "Biological Sciences",
    "CHM": "Chemical Sciences", "CUAB-BCH": "Biochemistry", "CUAB": "Crescent University - General"
}

# Text preprocessing
def normalize_text(text):
    text = re.sub(r'([^a-zA-Z0-9\s])', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    return text

def preprocess_text(text):
    text = normalize_text(text)
    words = text.split()
    expanded = [abbreviations.get(word.lower(), word) for word in words]
    corrected = []
    for word in expanded:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        corrected.append(suggestions[0].term if suggestions else word)
    return ' '.join(corrected)

def extract_prefix(code):
    match = re.match(r"([A-Z\-]+)", code)
    return match.group(1) if match else None

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    with open("qa_dataset.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    rag_data = []
    for entry in raw_data:
        if entry.get("question") and entry.get("answer"):
            rag_data.append({
                "text": f"Q: {entry['question'].strip()}\nA: {entry['answer'].strip()}",
                "question": entry["question"].strip(),
                "answer": entry["answer"].strip(),
                "department": entry.get("department", "").strip(),
                "level": entry.get("level", "").strip(),
                "semester": entry.get("semester", "").strip(),
                "faculty": entry.get("faculty", "").strip()
            })

    return pd.DataFrame(rag_data)

# Build a map from course titles to course info (code, department)
def build_title_map(dataset):
    title_map = {}
    for _, row in dataset.iterrows():
        question = row['question'].lower()
        # Try to find course code like MTH101 or MTH-101
        match = re.search(r"([A-Z]{2,}-?\d{3,})", row['question'])
        if match:
            code = match.group(1)
            prefix = extract_prefix(code)
            department = department_map.get(prefix, "Unknown")
            # Use question text (or a shortened version) as key, lowercase for matching
            key = question
            title_map[key] = {
                "code": code,
                "department": department,
                "answer": row['answer']
            }
    return title_map

@st.cache_data
def compute_question_embeddings(questions: list):
    model = load_model()
    return model.encode(questions, convert_to_tensor=True)

# GPT fallback with dynamic context
def fallback_openai(user_input, dataset, max_context=10):
    system_prompt = (
        "You are a helpful assistant specialized in Crescent University information. "
        "Use the provided Q&A examples to answer the question. If you are unsure, say so and refer the user to official sources."
    )
    messages = [{"role": "system", "content": system_prompt}]

    context_examples = dataset.sample(n=min(max_context, len(dataset))).to_dict("records")
    context_text = "\n".join([f"Q: {row['question']}\nA: {row['answer']}" for row in context_examples])

    messages.append({"role": "user", "content": f"Here is some university information:\n{context_text}"})
    messages.append({"role": "user", "content": f"Answer this question: {user_input}"})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.3
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        print("OpenAI Error:", e)
        return "Sorry, I couldn't reach the server. Try again later."

# Full response finder with dynamic context
def find_response(user_input, dataset, embeddings, threshold=0.4):
    model = load_model()
    user_input_clean = preprocess_text(user_input)

    # Check if user asks specifically about a course and department
    if "which course is" in user_input_clean.lower() and "department" in user_input_clean.lower():
        for title, info in title_map.items():
            if title in user_input_clean.lower():
                return (
                    f"The course titled '{title.title()}' is {info['code']}, and it is offered by the {info['department']} department.",
                    info["department"],
                    1.0,
                    []
                )

    if embeddings is None or len(dataset) == 0:
        return "No matching data found for your filters.", None, 0.0, []

    greetings = ["hi", "hello", "hey", "hi there", "greetings", "how are you",
                 "how are you doing", "how's it going", "can we talk?",
                 "can we have a conversation?", "okay", "i'm fine", "i am fine"]
    if user_input_clean.lower() in greetings:
        return random.choice([
            "Hello!", "Hi there!", "Hey!", "Greetings!", "I'm doing well, thank you!",
            "Sure pal", "I'm fine, thank you", "Hi! How can I help you?",
            "Hello! Ask me anything about Crescent University."
        ]), None, 1.0, []

    user_embedding = model.encode(user_input_clean, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(user_embedding, embeddings)[0]
    top_scores, top_indices = torch.topk(cos_scores, k=5)

    top_score = top_scores[0].item()
    top_index = top_indices[0].item()

    if top_score < threshold:
        top_context = [dataset.iloc[i.item()] for i in top_indices[:5]]
        context_df = pd.DataFrame(top_context)
        gpt_reply = fallback_openai(user_input, context_df)
        return gpt_reply, None, top_score, []

    response = dataset.iloc[top_index]["answer"]
    question = dataset.iloc[top_index]["question"]
    related_questions = [dataset.iloc[i.item()]["question"] for i in top_indices[1:]]

    match = re.search(r"\b([A-Z]{2,}-?\d{3,})\b", question)
    department = None
    if match:
        code = match.group(1)
        prefix = extract_prefix(code)
        department = department_map.get(prefix, "Unknown")

    if random.random() < 0.2:
        response = random.choice(["I think ", "Maybe: ", "Possibly: ", "Here's what I found: "]) + response

    return response, department, top_score, related_questions

# --- Streamlit UI ---
st.set_page_config(page_title="Crescent University Chatbot", page_icon="🎓")

# --- Initialize session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "related_questions" not in st.session_state:
    st.session_state.related_questions = []

if "last_department" not in st.session_state:
    st.session_state.last_department = None

dataset = load_data()
model = load_model()

title_map = build_title_map(dataset)

question_list = dataset['question'].tolist()
question_embeddings = compute_question_embeddings(question_list)

# --- Apply filters ---
def apply_filters(df, faculty, department, level, semester):
    filtered_df = df.copy()
    if faculty:
        filtered_df = filtered_df[filtered_df['faculty'].isin(faculty)]
    if department:
        filtered_df = filtered_df[filtered_df['department'].isin(department)]
    if level:
        filtered_df = filtered_df[filtered_df['level'].isin(level)]
    if semester:
        filtered_df = filtered_df[filtered_df['semester'].isin(semester)]
    return filtered_df

# --- Sidebar Filters ---
with st.sidebar:
    st.header("Filter Questions")

    faculties = sorted(dataset['faculty'].dropna().unique().tolist())
    departments = sorted(dataset['department'].dropna().unique().tolist())
    levels = sorted(dataset['level'].dropna().unique().tolist())
    semesters = sorted(dataset['semester'].dropna().unique().tolist())

    selected_faculty = st.multiselect("Faculty", faculties)
    selected_department = st.multiselect("Department", departments)
    selected_level = st.multiselect("Level", levels)
    selected_semester = st.multiselect("Semester", semesters)

    if st.button("Clear Chat"):
        st.session_state.chat_history.clear()
        st.session_state.related_questions.clear()
        st.session_state.last_department = None
        st.experimental_rerun()

filtered_dataset = apply_filters(dataset, selected_faculty, selected_department, selected_level, selected_semester)
filtered_questions = filtered_dataset['question'].tolist()
filtered_embeddings = compute_question_embeddings(filtered_questions) if filtered_questions else None

# --- Main Chat UI ---
st.title("🎓 Crescent University Chatbot")
user_input = st.text_input("Ask me anything about Crescent University...")

if user_input:
    answer, department, confidence, related_qs = find_response(
        user_input,
        filtered_dataset if (selected_faculty or selected_department or selected_level or selected_semester) else dataset,
        filtered_embeddings if filtered_embeddings is not None else question_embeddings
    )

    st.session_state.chat_history.append({"user": user_input, "bot": answer})
    st.session_state.related_questions = related_qs
    st.session_state.last_department = department

# --- Display Chat History ---
for chat in st.session_state.chat_history:
    st.markdown(f"You: {chat['user']}")
    st.markdown(f"Bot: {chat['bot']}")

# --- Display Related Questions ---
if st.session_state.related_questions:
    st.markdown("---")
    st.markdown("Related Questions:")
    for q in st.session_state.related_questions:
        st.write(f"- {q}")
