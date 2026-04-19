"""
train.py - The Strategy Sandbox Runner
This script runs the strategy defined in strategy.py.
"""
from prepare import evaluate_strategy

if __name__ == "__main__":
    # Evaluate strategy for 1-year period (matches prepare.py's default)
    final_score = evaluate_strategy('1y')
    
    # Output the exact string the AI is looking for
    print(f"FINAL_RESULT:{final_score}")
