import sys
import os
# Force Python to look in the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy_trainer.rag_memory import StrategyMemoryBank

def run_rag_test():
    print("🚀 Initializing ChromaDB Memory Bank...")
    # Use a separate test folder so we don't pollute your real database
    memory_bank = StrategyMemoryBank(db_path="./test_chroma_db") 
    
    print("\n📝 Injecting mock trials...")
    memory_bank.log_trial("a1b2c", "Added a tight trailing ATR stop loss of 1.5x", -0.05, "discard")
    memory_bank.log_trial("d4e5f", "Used CVD and Open Interest to filter out sideways chop", 0.12, "keep")
    memory_bank.log_trial("g7h8i", "Removed RSI and added Bollinger Bands mean reversion", -0.15, "discard")
    
    print(f"📊 Total memories in DB: {memory_bank.collection.count()}")

    print("\n🔍 AI wants to try: 'I want to add a dynamic ATR stop loss'")
    results = memory_bank.query_similar_trials("I want to add a dynamic ATR stop loss", n_results=2)
    
    print("\n🧠 Top Context Results:")
    for i, r in enumerate(results):
        print(f"{i+1}. {r['summary']} | Score: {r['score']} | Status: {r['status']}")

if __name__ == "__main__":
    run_rag_test()