from dotenv import load_dotenv
import os
from google import genai

import chromadb
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="db2_docs")
from pypdf import PdfReader
pdf_files = ["Db2Doc.pdf", "DB2BACKUP.pdf", "DB2RESTORE.pdf"]
full_text = ""
for pdf_file in pdf_files:
    reader = PdfReader(pdf_file)
    print("Total pages in", pdf_file, ":", len(reader.pages))
    for page in reader.pages:
        full_text += page.extract_text() + "\n"

print("Total characters extracted:", len(full_text))
print("\nFirst 300 characters:\n", full_text[:300])

#Chunking the text into smaller parts for better processing
chunk_size = 400  # Adjust this size based on your needs
chunks = []
overlap = 100    # Overlap between chunks to maintain context
for i in range(0, len(full_text), chunk_size - overlap):
    chunk = full_text[i:i+chunk_size]
    chunks.append(chunk)

print("Total chunks created:", len(chunks))
print("\nFirst chunk:\n", chunks[0])

collection.upsert(
    documents=chunks,
    ids=[f"id_{i+1}" for i in range(len(chunks))]
)

print("Chunks stored in Chroma:", collection.count())

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

while True:
    user_question = input("Enter your question (type 'exit' to quit): ")
    #print(f"DEBUG - you typed: [{user_question}]")
    if user_question.lower() == 'exit':
        break
    results = collection.query(
        query_texts=[user_question],
        n_results=3

)
    

    print("Retrieved chunks for query:", user_question)
    for retrieved_chunk in results['documents'][0]:
        print("\n---")
        print(retrieved_chunk)

    
    context = "\n".join(results['documents'][0])

    prompt = f"""Extracted from a Db2 document: {context}
    Based on this context, answer the question: {user_question}"""


    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    print("\nGemini's answer:")
    print(response.text)