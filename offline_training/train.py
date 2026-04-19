"""
train.py - The Strategy Sandbox
The AI agent edits this file to maximize FINAL_RESULT.
"""
import pandas as pd
from prepare import fetch_market_data, evaluate_risk

def run_strategy(df):
    """
    The trading strategy logic. 
    The AI should modify indicators and rules here.
    """
    equity = 10000.0
    equity_curve = [equity]
    
    # Simple dummy strategy for the baseline
    for i in range(1, len(df)):
        # AI will replace this with real RSI/MACD/Price Action logic
        trade_result = (df['close'].iloc[i] - df['close'].iloc[i-1]) 
        equity += trade_result
        equity_curve.append(equity)
        
    return equity_curve

if __name__ == "__main__":
    # 1. Get the data
    df = pd.read_csv("C:/Users/jeand/Desktop/Crypto_LLM/data/btc_1h_1y.csv")
    
    # 2. Run the strategy
    equity_curve = run_strategy(df)
    
    # 3. The Referee evaluates the result
    final_score = evaluate_risk(equity_curve)
    
    # 4. Output the exact string the AI is looking for
    print(f"FINAL_RESULT:{final_score}")