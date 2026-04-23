import chromadb

def check_database_size():
    try:
        # Connect to your running ChromaDB Server
        client = chromadb.HttpClient(host="localhost", port=8000)
        
        # Grab your specific collection
        collection = client.get_collection(name="strategy_trials")
        
        # Count the total number of items stored
        total_rows = collection.count()
        
        print(f"🧠 Total hypotheses logged in ChromaDB: {total_rows}")
        
    except Exception as e:
        print(f"⚠️ Could not connect. Is the Chroma server running? Error: {e}")

if __name__ == "__main__":
    check_database_size()