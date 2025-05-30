import os
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import streamlit as st
load_dotenv()


# Load Gemini API
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.0-flash-exp")

# Embedding model for semantic search
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

def find_relevant_claims(df, query, top_n=5):
    corpus = df["cleaned"].astype(str).tolist()
    query_emb = sentence_model.encode(query, convert_to_tensor=True)
    doc_embs = sentence_model.encode(corpus, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_emb, doc_embs)[0]
    top_indices = similarities.argsort(descending=True)[:top_n]
    return df.iloc[top_indices.tolist()]

def format_context_rows(rows: pd.DataFrame) -> str:
    lines = []
    for _, row in rows.iterrows():
        line = f"Claim ID: {row['claim_id']}, Category: {row['category']}, Specialty: {row['specialty']}, Insurance: {row['insurance_type']}, Comment: {row['cleaned']}, Sentiment: {row['Sentiment']}, Expected: {row['amount_expected']}, Paid: {row['amount_paid']}"
        lines.append(line)
    return "\n".join(lines)

def build_prompt(user_question: str, claims_context: str) -> str:
    return f"""
You are an expert medical claim analyst AI. Your task is to help users understand healthcare claims based on patterns and reasons. 
Analyze the following claims:

{claims_context}

User Question: "{user_question}"

Respond with a clear, human-like explanation, summarizing what patterns or reasons explain this. Only refer to trends from the data above. Suggest actionable next steps if appropriate.
"""

def gpt_response(df: pd.DataFrame, user_query: str) -> str:
    try:
        relevant_claims = find_relevant_claims(df, user_query, top_n=6)
        context = format_context_rows(relevant_claims)
        prompt = build_prompt(user_query, context)

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"❌ GPT Error: {e}"
