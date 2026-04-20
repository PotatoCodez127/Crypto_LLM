"""
train.py - The Strategy Sandbox
"""
from prepare import evaluate_strategy
if __name__ == "__main__":
    final_score = evaluate_strategy('1y')
    print(f"FINAL_RESULT:{final_score}")