import ollama
from dotenv import load_dotenv
import os
from google import genai

import chromadb
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="db2_docs")
from pypdf import PdfReader

# ==================== INDEXING PIPELINE ====================
pdf_files = ["Db2Doc.pdf", "DB2BACKUP.pdf", "DB2RESTORE.pdf"]
full_text = ""
for pdf_file in pdf_files:
    reader = PdfReader(pdf_file)
    print("Total pages in", pdf_file, ":", len(reader.pages))
    for page in reader.pages:
        full_text += page.extract_text() + "\n"

print("Total characters extracted:", len(full_text))
print("\nFirst 300 characters:\n", full_text[:300])

# ==================== CHUNKING (LangChain) ====================
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100
)
chunks = splitter.split_text(full_text)

print("Total chunks created:", len(chunks))
print("\nFirst chunk:\n", chunks[0])

collection.upsert(
    documents=chunks,
    ids=[f"id_{i+1}" for i in range(len(chunks))]
)

print("Chunks stored in Chroma:", collection.count())

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ==================== BM25 SETUP ====================
from rank_bm25 import BM25Okapi
tokenized_chunks = [chunk.split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)

# ==================== RETRIEVAL + GENERATION (Interactive Loop) ====================
while True: 
    user_question = input("Enter your question (type 'exit' to quit): ")
    if user_question.lower() == 'exit':
        break
    results = collection.query(
        query_texts=[user_question],
        n_results=3
    )

    tokenized_query = user_question.split()
    bm25_top_chunks = bm25.get_top_n(tokenized_query, chunks, n=3)
    
    semantic_chunks = results['documents'][0]
    combined_chunks = semantic_chunks + bm25_top_chunks
    combined_chunks = list(set(combined_chunks))
    
    print("Retrieved chunks for query:", user_question)
    for retrieved_chunk in combined_chunks:
        print("\n---")
        print(retrieved_chunk)

    context = "\n".join(combined_chunks)

    prompt = f"""Extracted from a Db2 document: {context}
    Based on this context, answer the question: {user_question}"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    ollama_response = ollama.chat(
        model='llama3.2',
        messages=[{'role': 'user', 'content': prompt}]
    )
    
    print("\nOllama's answer:")
    print(ollama_response['message']['content'])

    print("\nGemini's answer:")
    print(response.text)

# ==================== EVALUATION ====================
test_questions = [
    "db2 crash issue, what should I check?",
    "How to backup db2 database?",
    "How to restore db2 database?"
]

expected_keywords = [
    ["db2", "crash", "issue", "check"],
    ["backup", "db2", "database"],
    ["restore", "db2", "database"]
]

print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)

for question, keywords in zip(test_questions, expected_keywords):
    eval_results = collection.query(query_texts=[question], n_results=3)
    retrieved_text = " ".join(eval_results['documents'][0]).lower()
    
    matched = [kw for kw in keywords if kw.lower() in retrieved_text]
    
    if len(matched) == len(keywords):
        print(f"PASS: '{question}' -> all keywords found: {keywords}")
    else:
        print(f"PARTIAL/FAIL: '{question}' -> found {matched}, missing {[kw for kw in keywords if kw not in matched]}")



test_response = ollama.chat(
    model='llama3.2',
    messages=[{'role': 'user', 'content': 'What is a database backup?'}]
)

print(test_response['message']['content'])