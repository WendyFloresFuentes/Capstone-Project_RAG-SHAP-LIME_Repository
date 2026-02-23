# -*- coding: utf-8 -*-

# =========================
# IMPORTS
# =========================
import streamlit as st
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict

from PyPDF2 import PdfReader
from openai import OpenAI

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

import shap
from lime.lime_text import LimeTextExplainer

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Trustworthy AI Explainer",
    page_icon="🤖",
    layout="wide"
)

# =========================
# LLM
# =========================
@st.cache_resource
def load_llm():
    return OpenAI()

# =========================
# SESSION STATE
# =========================
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "vectordb" not in st.session_state:
        st.session_state.vectordb = None

    if "current_explanation" not in st.session_state:
        st.session_state.current_explanation = None

    if "preferences" not in st.session_state:
        st.session_state.preferences = {
            "temperature": 0.7,
            "max_tokens": 500
        }

    if "feedback_db" not in st.session_state:
        st.session_state.feedback_db = []

# =========================
# PDF → VECTOR DB
# =========================
@st.cache_resource
def process_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)

    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.create_documents([text])

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(docs, embeddings)

    return vectordb

# =========================
# RAG + LLM
# =========================
def generate_response(message: str, temperature: float):
    start = time.time()
    client = load_llm()

    if st.session_state.vectordb is None:
        return "Please upload a PDF first.", None, 0.0, []

    retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(message)

    chunks = [d.page_content for d in docs]
    context = "\n\n".join(chunks) if chunks else ""

    system_prompt = (
        "You are a strict AI assistant.\n"
        "Answer ONLY using the context.\n"
        "If the answer is not in the context, say:\n"
        "'I cannot find that information in the provided document.'"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{message}"
        }
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=temperature,
        max_tokens=st.session_state.preferences["max_tokens"]
    )

    output = response.choices[0].message.content
    elapsed = time.time() - start

    explanation = {
        "confidence": 0.85,
        "top_features": ["Context relevance", "Semantic similarity"]
    }

    return output, explanation, elapsed, chunks

# =========================
# SHAP
# =========================
def shap_explanation(chunks: List[str], question: str):
    embeddings = OpenAIEmbeddings()
    chunk_embs = embeddings.embed_documents(chunks)
    q_emb = embeddings.embed_query(question)

    def model(x):
        return np.array([
            np.dot(emb, q_emb) for emb in chunk_embs
        ])

    explainer = shap.Explainer(model, chunks)
    return explainer(chunks)

# =========================
# LIME
# =========================
def lime_explanation(chunks: List[str], question: str):
    explainer = LimeTextExplainer(class_names=["relevant"])
    embeddings = OpenAIEmbeddings()
    q_emb = embeddings.embed_query(question)

    def predictor(texts):
        scores = []
        for t in texts:
            t_emb = embeddings.embed_query(t)
            scores.append([np.dot(t_emb, q_emb)])
        return np.array(scores)

    return explainer.explain_instance(
        " ".join(chunks),
        predictor,
        num_features=5
    )

# =========================
# UI
# =========================
def page_chat():
    st.title("💬 RAG + SHAP + LIME")

    with st.sidebar:
        st.header("📄 Upload PDF")
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

        if uploaded_file:
            with st.spinner("Processing PDF..."):
                st.session_state.vectordb = process_pdf(uploaded_file)
            st.success("PDF indexed")

    col1, col2 = st.columns([2, 1])

    with col1:
        if prompt := st.chat_input("Ask a question"):
            response, explanation, rt, chunks = generate_response(
                prompt,
                st.session_state.preferences["temperature"]
            )

            st.session_state.current_explanation = {
                "input": prompt,
                "output": response,
                "details": explanation,
                "response_time": rt,
                "chunks": chunks
            }

            st.chat_message("assistant").markdown(response)

    with col2:
        st.subheader("Explainability")

        if st.session_state.current_explanation:
            exp = st.session_state.current_explanation
            chunks = exp["chunks"]

            st.metric("Confidence", exp["details"]["confidence"])
            st.metric("Response time", f"{exp['response_time']:.2f}s")

            if chunks:
                st.subheader("🧠 LIME")
                lime_exp = lime_explanation(chunks, exp["input"])
                st.components.v1.html(lime_exp.as_html(), height=300)

                st.subheader("📊 SHAP")
                shap_values = shap_explanation(chunks, exp["input"])
                st.pyplot(shap.plots.bar(shap_values))

# =========================
# MAIN
# =========================
def main():
    initialize_session_state()
    page_chat()

if __name__ == "__main__":
    main()
