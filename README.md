# 📄🤖 Oracle RAG Assistant  
A Retrieval-Augmented Generation (RAG) application powered by **Oracle AI Vector Database**, **OpenAI**, and **LangChain**.  
Users can upload a **PDF** or scrape a **webpage**, store the extracted content in a vector database, and ask natural language questions with accurate, source-grounded answers.

---

## 🚀 Features

### 🧩 Multi-Source Document Ingestion
- 📄 **PDF ingestion** (automatic text extraction & chunking)
- 🌐 **Webpage ingestion** (URL scraping and content extraction)

### 🧠 AI-Powered Search
- Embeds text using **OpenAI text-embedding-3-small**
- Stores embeddings in **Oracle AI Vector Search**
- Fast semantic retrieval using **HNSW vector index**

### 🤖 RAG Question Answering
- Retrieves the most relevant chunks from Oracle
- Sends context + question to **GPT-4.1-mini**
- Produces grounded, factual answers (no hallucinations)

### 🖥 Clean Streamlit UI
- PDF upload panel  
- Webpage URL ingestion  
- Query interface  
- Status messages & progress indicators  

---

## 🏗️ Architecture Overview

