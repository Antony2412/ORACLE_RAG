# rag_oracle.py
import os
from typing import List

import oracledb
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from langchain_oracledb.vectorstores import oraclevs
from langchain_oracledb.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.document_loaders import WebBaseLoader


load_dotenv()

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_CHAT_MODEL = "gpt-4.1-mini"


def get_oracle_connection():
    user = os.getenv("ORACLE_USER")
    pwd = os.getenv("ORACLE_PASSWORD")
    dsn = os.getenv("ORACLE_DSN")

    if not all([user, pwd, dsn]):
        raise RuntimeError("ORACLE_USER, ORACLE_PASSWORD, ORACLE_DSN must be set in .env")

    conn = oracledb.connect(user=user, password=pwd, dsn=dsn)
    return conn


def get_embeddings():
    return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)


def get_chat_model():
    return ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)


def pdf_to_documents(pdf_path: str) -> List[Document]:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )

    docs = splitter.split_documents(pages)
    for i, d in enumerate(docs):
        d.metadata = d.metadata or {}
        d.metadata["chunk_id"] = i
    return docs

def url_to_documents(url: str):
    """Load webpage content and split into chunks."""
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )

    docs = splitter.split_documents(docs)

    # Add metadata
    for i, d in enumerate(docs):
        d.metadata["source_url"] = url
        d.metadata["chunk_id"] = i

    return docs    


def build_oracle_vector_store(docs: List[Document]):
    table_name = os.getenv("ORACLE_VECTOR_TABLE", "RAG_PDF_CHUNKS")

    conn = get_oracle_connection()
    embeddings = get_embeddings()

    vector_store = OracleVS.from_documents(
        docs,
        embeddings,
        client=conn,
        table_name=table_name,
        distance_strategy=DistanceStrategy.COSINE,
    )

    return conn, vector_store


def create_vector_index(conn, vector_store):
    idx_name = os.getenv("ORACLE_VECTOR_INDEX", "HNSW_RAG_PDF_IDX")

    oraclevs.create_index(
        conn,
        vector_store,
        params={
            "idx_name": idx_name,
            "idx_type": "HNSW",
        },
    )


def similarity_search(vector_store, query: str, k: int = 4) -> List[Document]:
    return vector_store.similarity_search(query, k)


def build_rag_prompt(query: str, docs: List[Document]) -> str:
    context = "\n\n---\n\n".join(
        f"Chunk {d.metadata.get('chunk_id', '')}:\n{d.page_content}" for d in docs
    )

    prompt = f"""
You are a helpful assistant answering questions only from the provided context.

Context:
{context}

Question: {query}

Instructions:
- Answer based ONLY on the context.
- If the answer is not in the context, say "I couldn't find this in the document."
- Be concise and clear.
    """.strip()

    return prompt


def answer_question(vector_store, query: str) -> str:
    docs = similarity_search(vector_store, query, k=4)
    prompt = build_rag_prompt(query, docs)

    llm = get_chat_model()
    response = llm.invoke(prompt)

    return response.content
