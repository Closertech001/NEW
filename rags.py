import streamlit as st
import re
import time
import random
import json
import os
from sentence_transformers import SentenceTransformer, util
from symspellpy.symspellpy import SymSpell
import pkg_resources
import openai

# --------------------------
# Load Q&A dataset
@st.cache_resource
def load_dataset():
    with open("qa_dataset.json", "r") as f:
        return json.load(f)

# --------------------------
# Initialize SymSpell
@st.cache_resource
def init_symspell():
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dict_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
    return sym_spell

# --------------------------
# Load SentenceTransformer model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------
# OpenAI GPT fallback
def gpt_fallback_response(prompt):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for Crescent University."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )
    return response["choices"][0]["message"]["content"]

# --------------------------
# Text Normalization
ABBREVIATIONS = {
    "u": "you", "r": "are", "ur": "your", "cn": "can", "cud": "could", "shud": "should", "wud": "would",
    "abt": "about", "bcz": "because", "plz": "please", "pls": "please", "tmrw": "tomorrow", "wat": "what",
    "wats": "what is", "info": "information", "yr": "year", "sem": "semester", "admsn": "admission",
    "clg": "college", "sch": "school", "uni": "university", "cresnt": "crescent", "l": "level", 
    "d": "the", "msg": "message", "idk": "i don't know", "imo": "in my opinion", "asap": "as soon as possible",
    "dept": "department", "reg": "registration", "fee": "fees", "pg": "postgraduate", "app": "application",
    "req": "requirement", "nd": "national diploma", "a-level": "advanced level", "alevel": "advanced level",
    "2nd": "second", "1st": "first", "nxt": "next", "prev": "previous", "exp": "experience",
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

def normalize_text(text, sym_spell):
    text = text.lower()
    for abbr, full in ABBREVIATIONS.items():
        text = re.sub(rf'\b{re.escape(abbr)}\b', full, text)
    for syn, rep in SYNONYMS.items():
        text = re.sub(rf'\b{re.escape(syn)}\b', rep, text)
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    return suggestions[0].term if suggestions else text

# --------------------------
# Greetings/Farewells
def is_greeting(text):
    return any(text.lower().strip() == g for g in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"])

def is_farewell(text):
    return any(text.lower().strip() == f for f in ["bye", "goodbye", "see you", "later", "farewell", "cya", "peace", "exit"])

def get_random_greeting_response():
    return random.choice([
        "Hello! How can I assist you today?",
        "Hi there! What can I help you with?",
        "Hey! Feel free to ask me anything about Crescent University.",
        "Greetings! How may I be of service?",
        "Hello! Ready to help you with any questions."
    ])

def get_random_farewell_response():
    return random.choice([
        "Goodbye! Have a great day!",
        "See you later! Feel free to come back anytime.",
        "Bye! Take care!",
        "Farewell! Let me know if you need anything else.",
        "Peace out! Hope to chat again soon."
    ])

# --------------------------
# Top-K Semantic Retrieval
def retrieve_top_k_answers(user_input, dataset, embed_model, top_k=3):
    user_embed = embed_model.encode(user_input, convert_to_tensor=True)
    questions = [item["question"] for item in dataset]
    q_embeds = embed_model.encode(questions, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(user_embed, q_embeds)[0]
    top_indices = scores.topk(top_k).indices
    results = []

    for idx in top_indices:
        idx = int(idx)
        results.append({
            "question": dataset[idx]["question"],
            "answer": dataset[idx]["answer"],
            "score": float(scores[idx])
        })

    return results

# --------------------------
# Follow-up Suggestions
def suggest_followups(top_question):
    tokens = top_question.split()
    if "admission" in tokens:
        return ["What are the admission requirements?", "Is there an admission fee?", "When is the deadline?"]
    elif "course" in tokens or "subject" in tokens:
        return ["What are the core courses?", "How many credit units?", "Who teaches this course?"]
    elif "fees" in tokens:
        return ["How much is tuition?", "Is there an installment plan?", "Are scholarships available?"]
    else:
        return ["Can you explain further?", "Where can I find more info?", "Who can I contact for help?"]

# --------------------------
# Main App
def main():
    st.set_page_config(page_title="Crescent University Chatbot", page_icon="🎓")
    st.title("🎓 Crescent University Chatbot")
    st.markdown("Ask me anything about your department, courses, or the university.")

    dataset = load_dataset()
    embed_model = load_embedding_model()
    sym_spell = init_symspell()

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your Crescent University assistant. Ask me anything!"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Type your question here...")

    if user_input:
        norm_input = normalize_text(user_input, sym_spell)

        if is_greeting(norm_input):
            response = get_random_greeting_response()
        elif is_farewell(norm_input):
            response = get_random_farewell_response()
        else:
            results = retrieve_top_k_answers(norm_input, dataset, embed_model)
            top_result = results[0]

            if top_result["score"] < 0.6:
                response = gpt_fallback_response(user_input)
            else:
                response = "Here are the top results I found:\n\n"
                for i, res in enumerate(results, 1):
                    response += f"**Option {i}:**\n**Q:** {res['question']}\n**A:** {res['answer']}\n\n"
                suggestions = suggest_followups(results[0]["question"])
                response += f"**You might also ask:**\n- " + "\n- ".join(suggestions)

        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").markdown(user_input)

        with st.chat_message("assistant"):
            typing_placeholder = st.empty()
            typing_placeholder.markdown("_Bot is typing..._")
            time.sleep(1.5)
            typing_placeholder.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
