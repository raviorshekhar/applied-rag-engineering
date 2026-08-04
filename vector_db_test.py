import chromadb
chroma_client = chromadb.Client()

# switch \`create_collection\` to \`get_or_create_collection\` to avoid creating a new collection every time
collection = chroma_client.get_or_create_collection(name="my_collection")

# switch \`add\` to \`upsert\` to avoid adding the same documents every time
collection.upsert(
    documents=[
        "db2 database is down",
            "db2 instance is down",
            "db2 server is down",
            "server's firewall is corrupted",
            "Kohli hits century",
            "Ronaldo is a GOAT"
    ],
    ids=["id1", "id2", "id3", "id4", "id5", "id6"]
)

results = collection.query(
    query_texts=["db2 crash issue"], # Chroma will embed this for you
    n_results=2 # how many results to return
)

print(results)