# utils/rag_engine.py

import json
import os
import random
import re

import pandas as pd
import streamlit as st
import torch

from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

from .preprocessing import preprocess_text, extract_prefix
from .constants import department_map


# --- Load Model ---
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# --- Load Data ---
@st.cache_data
def load_data():
    with open("qa_dataset.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    rag_data = []

    for entry in raw_data:
        question = entry.get("question")
        answer = entry.get("answer")

        if question and answer:
            question = str(question).strip()
            answer = str(answer).strip()

            if not question or not answer:
                continue

            rag_data.append({
                "text": f"Q: {question}\nA: {answer}",
                "question": question,
                "answer": answer,
                "department": entry.get("department", ""),
                "level": entry.get("level", ""),
                "semester": entry.get("semester", ""),
                "faculty": entry.get("faculty", ""),
            })

    return pd.DataFrame(
        rag_data,
        columns=[
            "text", "question", "answer", "department",
            "level", "semester", "faculty"
        ],
    )


# --- Compute Embeddings ---
def compute_question_embeddings(questions: list):
    model = load_model()

    processed_questions = [
        preprocess_text(str(question))
        for question in questions
    ]

    if not processed_questions:
        return torch.empty((0, 384))

    return model.encode(
        processed_questions,
        convert_to_tensor=True,
        show_progress_bar=False,
    )


# --- Fallback with OpenAI ---
def fallback_openai(user_input, context_qa=None):
    system_prompt = (
        "You are CUAB Buddy, a helpful assistant specializing in "
        "Crescent University, Nigeria. Use reliable university "
        "information when available. Never invent course codes, fees, "
        "admission requirements, or university policies. If you do not "
        "know an answer, say so politely and recommend checking "
        "official university sources."
    )

    messages = [{"role": "system", "content": system_prompt}]

    if context_qa:
        context_text = (
            "Potentially relevant university information:\n"
            f"Question: {context_qa['question']}\n"
            f"Answer: {context_qa['answer']}\n\n"
            "Use this information only if it is relevant to the question. "
        )
        user_message = (
            context_text + "User's question: " + str(user_input)
        )
    else:
        user_message = str(user_input)

    messages.append({"role": "user", "content": user_message})

    try:
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            try:
                api_key = st.secrets.get("OPENAI_API_KEY")
            except Exception:
                api_key = None

        client = OpenAI(api_key=api_key) if api_key else OpenAI()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
        )

        answer = response.choices[0].message.content

        if answer and answer.strip():
            return answer.strip()

        return "Sorry, I couldn't generate an answer. Please try again."

    except Exception as error:
        print(f"OpenAI API error: {error}")
        return (
            "Sorry, I'm having trouble connecting to the AI service "
            "right now. Please try again shortly."
        )


# --- Main RAG Response Logic ---
def find_response(user_input, dataset, embeddings, threshold=0.4):
    if dataset is None or dataset.empty:
        return (
            "I don't have university information available to search yet.",
            None,
            0.0,
            [],
        )

    if embeddings is None:
        return (
            "The university search index is not ready. Please try again later.",
            None,
            0.0,
            [],
        )

    if len(dataset) != len(embeddings):
        return (
            "The university search index needs to be rebuilt. "
            "Please try again later.",
            None,
            0.0,
            [],
        )

    model = load_model()
    user_input_clean = preprocess_text(str(user_input))

    greetings = {
        "hi",
        "hello",
        "hey",
        "hi there",
        "greetings",
        "how are you",
        "how are you doing",
        "how's it going",
        "can we talk",
        "can we have a conversation",
        "okay",
        "i'm fine",
        "i am fine",
    }

    if user_input_clean.lower().strip() in greetings:
        return random.choice([
            "Hello!",
            "Hi there!",
            "Hey!",
            "I'm doing well, thank you!",
            "Hi! How can I help you?",
            "Hello! Ask me anything about Crescent University.",
        ]), None, 1.0, []

    try:
        user_embedding = model.encode(
            user_input_clean,
            convert_to_tensor=True,
        )

        cos_scores = util.pytorch_cos_sim(
            user_embedding, embeddings
        )[0]

    except Exception as error:
        print(f"Retrieval error: {error}")
        return (
            "Sorry, something went wrong while searching "
            "university information.",
            None,
            0.0,
            [],
        )

    if cos_scores.numel() == 0:
        return (
            "I couldn't find university information to match your question yet.",
            None,
            0.0,
            [],
        )

    top_k = min(5, cos_scores.numel())
    top_scores, top_indices = torch.topk(cos_scores, k=top_k)

    top_score = top_scores[0].item()
    top_index = top_indices[0].item()

    if top_score < threshold:
        # Do not send a weak, potentially unrelated match as context.
        gpt_reply = fallback_openai(user_input)
        return gpt_reply, None, top_score, []

    response = dataset.iloc[top_index]["answer"]
    question = dataset.iloc[top_index]["question"]

    related_questions = [
        dataset.iloc[index.item()]["question"]
        for index in top_indices[1:]
    ]

    match = re.search(r"\b([A-Z]{2,}-?\d{3,})\b", question)
    department = None

    if match:
        code = match.group(1)
        prefix = extract_prefix(code)
        department = department_map.get(prefix, "Unknown")

    return response, department, top_score, related_questions
