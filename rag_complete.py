from dotenv import load_dotenv
import os
import chromadb
from google import genai
load_dotenv()

# Step 1: Set up Chroma client and collection (same as before)
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="my_collection")

collection.upsert(
    documents=[
        "db2 database is down",
        "db2 instance is down",
        "db2 server is down",
        "server's firewall is corrupted",
        "Kohli hits century",
        "Ronaldo is a GOAT",
        "To troubleshoot db2 instance down, check db2diag.log file and verify instance owner permissions"
    ],
    ids=["id1", "id2", "id3", "id4", "id5", "id6", "id7"]
)

# Step 2: Define the user's question
user_question = "db2 crash issue, what should I check?"

# Step 3: Retrieve the most relevant documents from Chroma
results = collection.query(
    query_texts=[user_question],
    n_results=3
)

retrieved_docs = results['documents'][0]
print("Retrieved documents:", retrieved_docs)

# Step 4: Combine retrieved documents into context
context = "\n".join(retrieved_docs)

# Step 5: Send context + question to Gemini to generate the final answer
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

prompt = f"""why db2 is down:
{context}

Based on this context, answer the question: {user_question}"""

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
)

print("\nGemini's answer:")
print(response.text)