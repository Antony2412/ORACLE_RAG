# app.py
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from rag_oracle import (
    pdf_to_documents,
    build_oracle_vector_store,
    create_vector_index,
    answer_question,
)

load_dotenv()

st.set_page_config(page_title="Oracle RAG PDF Assistant", layout="wide")

st.title("📄🔍 Oracle RAG PDF Assistant")
st.write("Upload a PDF, store it in Oracle Vector DB, then ask questions about it.")


with st.sidebar:
    st.header("Oracle & OpenAI config")
    st.text_input("Oracle user", os.getenv("ORACLE_USER"), disabled=True)
    st.text_input("Oracle DSN", os.getenv("ORACLE_DSN"), disabled=True)
    st.text_input("Vector table", os.getenv("ORACLE_VECTOR_TABLE", "RAG_PDF_CHUNKS"), disabled=True)
    st.text_input("OpenAI model (chat)", "gpt-4.1-mini", disabled=True)
    st.text_input("OpenAI model (embedding)", "text-embedding-3-small", disabled=True)
    st.markdown("---")
    st.caption("Update `.env` to change these values.")


if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "ingested" not in st.session_state:
    st.session_state.ingested = False

tab_ingest, tab_query = st.tabs(["📥 Ingest PDF", "❓ Ask questions"])


with tab_ingest:
    st.subheader("1. Upload & Ingest PDF into Oracle Vector DB")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        st.success(f"Selected file: {uploaded_file.name}")

        if st.button("🚀 Ingest into Oracle"):
            with st.spinner("Reading PDF, chunking, embedding, and storing in Oracle..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                docs = pdf_to_documents(tmp_path)

                st.write(f"Extracted and split into **{len(docs)}** chunks. Storing in Oracle...")

                conn, vector_store = build_oracle_vector_store(docs)
                create_vector_index(conn, vector_store)

                st.session_state.vector_store = vector_store
                st.session_state.ingested = True

                st.success("✅ Ingestion complete and vector index created.")


with tab_query:
    st.subheader("2. Ask a question about your PDF")

    if not st.session_state.ingested or st.session_state.vector_store is None:
        st.info("Please ingest a PDF first in the **Ingest PDF** tab.")
    else:
        user_q = st.text_input("Your question")

        if st.button("💬 Get answer") and user_q.strip():
            with st.spinner("Running semantic search in Oracle and querying OpenAI..."):
                try:
                    answer = answer_question(st.session_state.vector_store, user_q)
                    st.markdown("### Answer")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error during RAG query: {e}")
