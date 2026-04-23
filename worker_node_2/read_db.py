import chromadb

def read_database():
    try:
        # Connect to your running ChromaDB Server
        client = chromadb.HttpClient(host="localhost", port=8000)
        
        # Grab your specific collection
        collection = client.get_collection(name="strategy_trials")
        
        # Fetch all the data (we ask for the metadata and the text documents)
        results = collection.get(
            include=["metadatas", "documents"]
        )
        
        if not results or not results['metadatas']:
            print("📭 The database is currently empty.")
            return
            
        total_rows = len(results['metadatas'])
        print(f"📖 Found {total_rows} entries in the memory bank.\n")
        print("="*80)
        
        # Loop through every row and print it cleanly
        for i in range(total_rows):
            metadata = results['metadatas'][i]
            # The 'document' is where we saved the actual text of the hypothesis
            document = results['documents'][i] if results['documents'] else "No text found."
            
            score = metadata.get('score', 'N/A')
            status = metadata.get('status', 'N/A')
            commit = metadata.get('commit', 'N/A')
            
            # Add a visual tag so you can spot the winners
            tag = "🏆 WINNER" if status == "keep" else "☠️ LANDMINE"
            
            print(f"[{i+1}] {tag} | Score: {score}")
            print(f"Git Commit: {commit}")
            print(f"Hypothesis: {document}")
            print("-" * 80)
            
    except Exception as e:
        print(f"⚠️ Could not connect. Is the Chroma server running? Error: {e}")

if __name__ == "__main__":
    read_database()