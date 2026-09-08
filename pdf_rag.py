from dotenv import load_dotenv
import os
from google import genai

import chromadb
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="db2_docs")
from pypdf import PdfReader

# ==================== INDEXING PIPELINE ====================
pdf_files = ["Db2Doc.pdf", "DB2BACKUP.pdf", "DB2RESTORE.pdf"] #Indexing multiple pdf files
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
#chunks = []
new_chunks=[]
overlap = 100    # Overlap between chunks to maintain context
for i in range(0, len(full_text), chunk_size - overlap):
    chunk = full_text[i:i+chunk_size]
    new_chunks.append(chunk)

print("Total chunks created:", len(new_chunks))
print("\nFirst chunk:\n", new_chunks[0])

collection.upsert( #save embeddings in Chroma or vector database
    documents=new_chunks,
    ids=[f"id_{i+1}" for i in range(len(new_chunks))]
)

print("Chunks stored in Chroma:", collection.count()) #Indexing done, now we can query the vector database for relevant chunks based on user questions

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


#Langchaain Integration

from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100,


)

test_text = full_text
new_chunks = splitter.split_text(test_text)

print("Total chunks (Langchain):", len(new_chunks))
print("\nFirst chunk:\n", new_chunks[0])



# ==================== BM25 SETUP ====================
from rank_bm25 import BM25Okapi
tokenized_chunks = [chunk.split() for chunk in new_chunks]
bm25 = BM25Okapi(tokenized_chunks)

# ==================== RETRIEVAL ====================
#Retrieve relevant chunks based on user question and send to Gemini for answer generation
while True: 
    user_question = input("Enter your question (type 'exit' to quit): ")
    #print(f"DEBUG - you typed: [{user_question}]")
    if user_question.lower() == 'exit':
        break
    results = collection.query( #Retrieve completed here
        query_texts=[user_question],
        n_results=3

)

    tokenized_query = user_question.split()
    bm25_top_chunks = bm25.get_top_n(tokenized_query, new_chunks, n=3)
    
    semantic_chunks = results['documents'][0]
    combined_chunks = semantic_chunks + bm25_top_chunks
    combined_chunks = list(set(combined_chunks))
    
# ==================== GENERATION ====================
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

    print("\nGemini's answer:")
    print(response.text)


