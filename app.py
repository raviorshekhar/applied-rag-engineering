import streamlit as st
from dotenv import load_dotenv
import os
from google import genai
import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import ollama

# ==================== PAGE CONFIG (jhakkas look ke liए) ====================
st.set_page_config(
    page_title="DB2 Assistant",
    page_icon="🗄️",
    layout="centered"
)

# ==================== SETUP (runs once, cached) ====================
@st.cache_resource
def setup_rag():
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="db2_docs")

    pdf_files = ["Db2Doc.pdf", "DB2BACKUP.pdf", "DB2RESTORE.pdf"]
    full_text = ""
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            full_text += page.extract_text() + "\n"

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    chunks = splitter.split_text(full_text)

    collection.upsert(
        documents=chunks,
        ids=[f"id_{i+1}" for i in range(len(chunks))]
    )

    tokenized_chunks = [chunk.split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    load_dotenv()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    return collection, chunks, bm25, client

collection, chunks, bm25, client = setup_rag()

# ==================== UI ====================
st.title("🗄️ DB2 Assistant")
st.caption("Ask questions about DB2 troubleshooting, backup, and restore — grounded in official documentation.")

st.divider()

user_question = st.text_input("💬 Enter your question:", placeholder="e.g. How to restore a DB2 database?")

st.write("**Choose your model:**")
col1, col2 = st.columns(2)

with col1:
    gemini_clicked = st.button("☁️ Ask Gemini (Cloud)", use_container_width=True)

with col2:
    ollama_clicked = st.button("💻 Ask Ollama (Local)", use_container_width=True)

if (gemini_clicked or ollama_clicked) and user_question:
    with st.spinner("Retrieving relevant information..."):
        results = collection.query(query_texts=[user_question], n_results=3)
        tokenized_query = user_question.split()
        bm25_top_chunks = bm25.get_top_n(tokenized_query, chunks, n=3)

        semantic_chunks = results['documents'][0]
        combined_chunks = list(set(semantic_chunks + bm25_top_chunks))

        context = "\n".join(combined_chunks)
        prompt = f"""Extracted from a Db2 document: {context}
        Based on this context, answer the question: {user_question}"""

    if gemini_clicked:
        with st.spinner("Gemini is thinking..."):
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )
            answer = response.text
            model_used = "☁️ Gemini (Cloud)"
    else:
        with st.spinner("Ollama is thinking... (this may take a moment)"):
            ollama_client = ollama.Client(host='http://host.docker.internal:11434')
            ollama_response = ollama_client.chat(
                model='llama3.2',
                messages=[{'role': 'user', 'content': prompt}]
            )
            answer = ollama_response['message']['content']
            model_used = "💻 Ollama (Local)"

    st.divider()
    st.write(f"**Answered by {model_used}**")
    st.success(answer)

    with st.expander("📄 View retrieved source chunks"):
        for i, chunk in enumerate(combined_chunks):
            st.text_area(f"Chunk {i+1}", chunk, height=100)
elif (gemini_clicked or ollama_clicked) and not user_question:
    st.warning("Please enter a question first!")