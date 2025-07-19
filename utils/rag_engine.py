# utils/rag_engine.py
import json
import os
import random
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import openai
from .preprocessing import preprocess_text, extract_prefix
from .constants import department_map

openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Load Model ---
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# --- Load Data ---
def load_data():
    with open("qa_dataset.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    rag_data = []
    for entry in raw_data:
        if entry.get("question") and entry.get("answer"):
            rag_data.append({
                "text": f"Q: {entry['question'].strip()}\nA: {entry['answer'].strip()}",
                "question": entry['question'].strip(),
                "answer": entry['answer'].strip(),
                "department": entry.get("department", ""),
                "level": entry.get("level", ""),
                "semester": entry.get("semester", ""),
                "faculty": entry.get("faculty", "")
            })
    return pd.DataFrame(rag_data)

# --- Compute Embeddings ---
def compute_question_embeddings(questions: list):
    model = load_model()
    return model.encode(questions, convert_to_tensor=True)

# --- Fallback with OpenAI ---
def fallback_openai(user_input, context_qa=None):
    system_prompt = (
        "You are a helpful assistant specialized in Crescent University information. "
        "If you don't know an answer, politely say so and refer to university resources."
    )
    messages = [{"role": "system", "content": system_prompt}]

    if context_qa:
        context_text = f"Here is some relevant university information:\nQ: {context_qa['question']}\nA: {context_qa['answer']}\n\n"
        user_message = context_text + "Answer this question: " + user_input
    else:
        user_message = user_input

    messages.append({"role": "user", "content": user_message})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3
        )
        return response.choices[0].message["content"].strip()
    except Exception:
        return "Sorry, I couldn't reach the server. Try again later."

# --- Main RAG Response Logic ---
def find_response(user_input, dataset, embeddings, threshold=0.4):
    model = load_model()
    user_input_clean = preprocess_text(user_input)

    greetings = [
        "hi", "hello", "hey", "hi there", "greetings", "how are you",
        "how are you doing", "how's it going", "can we talk?",
        "can we have a conversation?", "okay", "i'm fine", "i am fine"
    ]
    if user_input_clean.lower() in greetings:
        return random.choice([
            "Hello!", "Hi there!", "Hey!", "Greetings!", "I'm doing well, thank you!", 
            "Sure pal", "I'm fine, thank you", "Hi! How can I help you?", 
            "Hello! Ask me anything about Crescent University."
        ]), None, 1.0, []

    try:
        user_embedding = model.encode(user_input_clean, convert_to_tensor=True)
    except Exception:
        return "Sorry, something went wrong while processing your question.", None, 0.0, []

    cos_scores = util.pytorch_cos_sim(user_embedding, embeddings)[0]
    top_scores, top_indices = torch.topk(cos_scores, k=5)

    top_score = top_scores[0].item()
    top_index = top_indices[0].item()

    if top_score < threshold:
        context_qa = {
            "question": dataset.iloc[top_index]["question"],
            "answer": dataset.iloc[top_index]["answer"]
        }
        gpt_reply = fallback_openai(user_input, context_qa)
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
