"""
rag_memory.py
Manages the local vector database (ChromaDB) to give the AI agent long-term memory.
"""
import os
import chromadb
from chromadb.utils import embedding_functions

class StrategyMemoryBank:
    def __init__(self):
        # Move the DB up one level so all worker clones can share it
        shared_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared_chroma_db"))
        self.client = chromadb.PersistentClient(path=shared_db_path)
        
        # We use Chroma's default lightweight local embedding model. 
        # (It will download a small ~80MB model on the very first run).
        self.emb_fn = embedding_functions.DefaultEmbeddingFunction()
        
        # Get or create our specific memory table
        self.collection = self.client.get_or_create_collection(
            name="strategy_trials",
            embedding_function=self.emb_fn
        )

    def log_trial(self, commit_hash: str, summary: str, score: float, status: str):
        """Saves a strategy attempt into the vector database."""
        self.collection.add(
            documents=[summary], # The text the AI will search against
            metadatas=[{"score": score, "status": status, "commit": commit_hash}], # Quantitative filters
            ids=[commit_hash] # Unique ID
        )
        print(f"🧠 Database Logged: [{commit_hash}] (Score: {score})")

    def query_similar_trials(self, current_idea: str, n_results: int = 3):
        """Retrieves the most semantically similar past trials based on the AI's current idea."""
        # Prevent querying if the DB is empty
        if self.collection.count() == 0:
            return []

        # Prevent crashing if we ask for 3 results but only 1 exists
        limit = min(n_results, self.collection.count())
        
        results = self.collection.query(
            query_texts=[current_idea],
            n_results=limit
        )
        
        memories = []
        # Parse the messy ChromaDB output into a clean list of dictionaries
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                meta = results['metadatas'][0][i]
                memories.append({
                    "summary": doc,
                    "score": meta["score"],
                    "status": meta["status"],
                    "commit": meta["commit"]
                })
        return memories

    def get_global_best_score(self):
        """Polls the shared ChromaDB to find the absolute highest score across all swarm workers."""
        try:
            # Fetch all successful trials from all workers
            results = self.collection.get(where={"status": "keep"})
            
            if not results or 'metadatas' not in results or not results['metadatas']:
                return -999.0
            
            # Extract scores and find the maximum
            scores = [float(meta['score']) for meta in results['metadatas'] if 'score' in meta]
            
            if scores:
                return max(scores)
            return -999.0
        except Exception as e:
            print(f"⚠️ Could not fetch global score: {e}")
            return -999.0