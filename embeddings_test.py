from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3 sentences - 2 with similar meaning and 1 different
sentences = [
    "db2 database is down",
    "db2 instance is down",
    "db2 server is down",
    "server's firewall is corrupted",
    "Kohli hits century",
    "Ronaldo is a GOAT",
]
sentence_embeddings = model.encode(sentences)
# Convert these into embeddings
#query = "database issue"
query = "cricket score"
query_embedding = model.encode(query)

for i in range(len(sentences)):
    score = cos_sim(query_embedding, sentence_embeddings[i])
    print(sentences[i], "->", score.item())


