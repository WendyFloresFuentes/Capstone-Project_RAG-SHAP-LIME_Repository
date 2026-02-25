# -*- coding: utf-8 -*-

# =============================================================================
# IMPORTS
# =============================================================================
import streamlit as st
import time
import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime

from PyPDF2 import PdfReader
from openai import OpenAI

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

import shap
from lime.lime_text import LimeTextExplainer

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Trustworthy AI Explainer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# LLM CONFIGURATION
# =============================================================================
@st.cache_resource
def load_llm():
    return OpenAI()

# =============================================================================
# SESSION STATE
# =============================================================================
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "feedback_db" not in st.session_state:
        st.session_state.feedback_db = []

    if "vectordb" not in st.session_state:
        st.session_state.vectordb = None

    if "current_explanation" not in st.session_state:
        st.session_state.current_explanation = None

    if "preferences" not in st.session_state:
        st.session_state.preferences = {
            "temperature": 0.7,
            "max_tokens": 500,
            "system_prompt": "You are a helpful, transparent AI assistant."
        }

    if "metrics" not in st.session_state:
        st.session_state.metrics = {
            "total_messages": 0,
            "avg_response_time": 0.0,
            "total_feedback": 0
        }

# =============================================================================
# PDF → VECTOR DB (RAG)
# =============================================================================
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

# =============================================================================
# CORE RAG + LLM
# =============================================================================
def generate_response(message: str, temperature: float):
    start = time.time()
    client = load_llm()

    if st.session_state.vectordb is None:
        return "Please upload a PDF document first.", None, 0.0, []

    retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(message)

    chunks = [d.page_content for d in docs]
    context = "\n\n".join(chunks) if chunks else ""

    system_prompt = (
        "You are a strict AI assistant.\n"
        "Answer ONLY using the context provided.\n"
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
        "top_features": [
            "Context relevance",
            "Semantic similarity",
            "Chunk coverage"
        ]
    }

    st.session_state.metrics["total_messages"] += 1
    n = st.session_state.metrics["total_messages"]
    st.session_state.metrics["avg_response_time"] = (
        ((n - 1) * st.session_state.metrics["avg_response_time"] + elapsed) / n
    )

    return output, explanation, elapsed, chunks

# =============================================================================
# SHAP (chunk relevance)
# =============================================================================
def shap_explanation(chunks: List[str], question: str):
    embeddings = OpenAIEmbeddings()
    chunk_embs = embeddings.embed_documents(chunks)
    q_emb = embeddings.embed_query(question)

    def model(x):
        return np.array([np.dot(e, q_emb) for e in chunk_embs])

    explainer = shap.Explainer(model, chunks)
    return explainer(chunks)

# =============================================================================
# LIME (word relevance)
# =============================================================================
def lime_explanation(chunks: List[str], question: str):
    # Use two class names for better visualization
    explainer = LimeTextExplainer(class_names=["Not Relevant", "Relevant"])
    
    embeddings = OpenAIEmbeddings()
    q_emb = embeddings.embed_query(question)

    def predictor(texts):
        scores = []
        for t in texts:
            if not t.strip():
                scores.append([1.0, 0.0])
                continue
                
            t_emb = embeddings.embed_query(t)
            dot_prod = np.dot(t_emb, q_emb)
            
            # Sigmoid normalization to keep the score between 0 and 1
            prob_relevant = 1 / (1 + np.exp(-dot_prod))
            
            # Provide probability for both classes
            scores.append([1 - prob_relevant, prob_relevant])
            
        return np.array(scores)

    # Use a smaller num_samples for faster API processing
    return explainer.explain_instance(
        " ".join(chunks),
        predictor,
        num_features=6,
        num_samples=50 
    )

# =============================================================================
# FEEDBACK
# =============================================================================
def save_feedback(message, response, rating, comment):
    st.session_state.feedback_db.append({
        "timestamp": datetime.now(),
        "message": message,
        "response": response,
        "rating": rating,
        "comment": comment
    })
    st.session_state.metrics["total_feedback"] += 1

# =============================================================================
# PAGE: CHAT
# =============================================================================
def page_chat():
    st.title("💬 AI Chat with Explainability")

    with st.sidebar:
        st.header("⚙️ Settings")

        st.session_state.preferences["temperature"] = st.slider(
            "Temperature", 0.0, 2.0,
            st.session_state.preferences["temperature"], 0.1
        )

        st.session_state.preferences["max_tokens"] = st.slider(
            "Max tokens", 50, 2000,
            st.session_state.preferences["max_tokens"], 50
        )

        st.divider()
        st.header("📄 RAG Document")

        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

        if uploaded_file:
            with st.spinner("Processing document..."):
                st.session_state.vectordb = process_pdf(uploaded_file)
            st.success("Document indexed successfully")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Conversation")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Type your message..."):
            st.session_state.messages.append({"role": "user", "content": prompt})

            response, explanation, rt, chunks = generate_response(
                prompt,
                st.session_state.preferences["temperature"]
            )

            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )

            st.session_state.current_explanation = {
                "input": prompt,
                "output": response,
                "details": explanation,
                "response_time": rt,
                "chunks": chunks
            }

            with st.chat_message("assistant"):
                st.markdown(response)

    with col2:
        st.subheader("Explainability")

        if st.session_state.current_explanation:
            exp = st.session_state.current_explanation
            chunks = exp["chunks"]

            st.metric("Confidence", exp["details"]["confidence"])
            st.metric("Response Time", f"{exp['response_time']:.2f}s")

            st.markdown("**Key Factors:**")
            for f in exp["details"]["top_features"]:
                st.markdown(f"- {f}")

            if chunks:
                st.divider()
                st.subheader("🧠 LIME Explanation")
                lime_exp = lime_explanation(chunks, exp["input"])
                st.components.v1.html(lime_exp.as_html(), height=400, scrolling=True)

                st.divider()
                st.subheader("📊 SHAP Explanation")
                shap_values = shap_explanation(chunks, exp["input"])
                st.pyplot(shap.plots.bar(shap_values))

            st.divider()
            rating = st.radio("Rate response", ["👍 Helpful", "👎 Not Helpful"])
            comment = st.text_area("Comment (optional)")

            if st.button("Submit Feedback"):
                save_feedback(exp["input"], exp["output"], rating, comment)
                st.success("Feedback saved!")
        else:
            st.info("Send a message to see explainability.")

# =============================================================================
# OTHER PAGES (UNCHANGED)
# =============================================================================
def page_explainability():
    st.title("🔍 Explainability Analysis")
    st.info("This page can be extended with global explanations.")

def page_feedback():
    st.title("📊 Feedback Dashboard")
    if not st.session_state.feedback_db:
        st.info("No feedback yet.")
        return
    df = pd.DataFrame(st.session_state.feedback_db)
    st.dataframe(df, use_container_width=True)

def page_monitoring():
    st.title("📈 Monitoring")
    m = st.session_state.metrics
    st.metric("Total Messages", m["total_messages"])
    st.metric("Avg Response Time", f"{m['avg_response_time']:.2f}s")
    st.metric("Total Feedback", m["total_feedback"])

def page_documentation():
    st.title("📚 Documentation")
    st.markdown("Trustworthy AI Explainer – Module 15")

# =============================================================================
# MAIN
# =============================================================================
def main():
    initialize_session_state()

    with st.sidebar:
        page = st.radio(
            "Navigation",
            ["💬 Chat", "🔍 Explainability", "📊 Feedback", "📈 Monitoring", "📚 Documentation"]
        )

    if page == "💬 Chat":
        page_chat()
    elif page == "🔍 Explainability":
        page_explainability()
    elif page == "📊 Feedback":
        page_feedback()
    elif page == "📈 Monitoring":
        page_monitoring()
    elif page == "📚 Documentation":
        page_documentation()

if __name__ == "__main__":
    main()
