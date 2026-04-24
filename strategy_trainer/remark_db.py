import chromadb

def salvage_runner_ups(margin=5.0, min_baseline=25.0):
    print("🔍 Connecting to ChromaDB Server...")
    try:
        # Connecting via HTTP as defined in your rag_memory.py architecture
        client = chromadb.HttpClient(host="localhost", port=8000)
        collection = client.get_collection(name="strategy_trials")
    except Exception as e:
        print(f"❌ Failed to connect to ChromaDB. Ensure your server is running. Error: {e}")
        return

    print("📦 Fetching historical trials...")
    results = collection.get(include=["metadatas"])
    
    if not results or not results['metadatas']:
        print("⚠️ Database is empty. Nothing to remark.")
        return

    ids = results['ids']
    metadatas = results['metadatas']
    
    # 1. Dynamically find the absolute global best score in the entire database
    all_scores = [float(meta.get('score', -999.0)) for meta in metadatas]
    global_best = max(all_scores) if all_scores else -999.0
    
    if global_best <= 0:
        print("⚠️ No profitable runs found to establish a baseline.")
        return
        
    print(f"🏆 Found Global High Score: {global_best}")
    print(f"🎯 Salvage Zone: Scores between {global_best - margin} and {global_best}")
    
    # 2. Iterate and update any discards that belong in the salvage zone
    updated_count = 0
    
    for i in range(len(ids)):
        doc_id = ids[i]
        meta = metadatas[i]
        score = float(meta.get('score', -999.0))
        status = meta.get('status', 'discard')
        
        if status == "discard":
            if score >= (global_best - margin) and score >= min_baseline:
                # Flip the status
                new_meta = meta.copy()
                new_meta['status'] = "keep"
                
                # Overwrite the row in ChromaDB
                collection.update(
                    ids=[doc_id],
                    metadatas=[new_meta]
                )
                updated_count += 1
                print(f"✅ Salvaged [{doc_id}] | Score: {score} -> Flipped to 'keep'")
                
    print(f"\n🎉 Finished! Successfully rescued {updated_count} high-performing runner-ups.")

if __name__ == "__main__":
    salvage_runner_ups()