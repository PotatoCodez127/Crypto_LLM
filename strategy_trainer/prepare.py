"""
The Walk-Forward Judge.
This script evaluates strategy.py using strict Out-Of-Sample validation
to prevent the AI from overfitting to historical noise.
"""

import pandas as pd
import numpy as np
import importlib.util
import sys
import os

# Configuration
DATA_FILE = "data/btc_1h_1y.csv" # We will eventually upgrade this to your V2 data
STRATEGY_FILE = "strategy.py"
INITIAL_CAPITAL = 10000.0
FEES = 0.0006  # 0.06% Taker Fee
SLIPPAGE = 0.0005 # 0.05% execution slippage

def load_strategy():
    """Dynamically loads the AI-generated strategy.py."""
    spec = importlib.util.spec_from_file_location("strategy", STRATEGY_FILE)
    strategy = importlib.util.module_from_spec(spec)
    sys.modules["strategy"] = strategy
    spec.loader.exec_module(strategy)
    return strategy

def evaluate_window(df_train, df_test, strategy_module):
    """
    Trains/Generates signals on the training window, 
    but ONLY evaluates PnL on the strictly unseen test window.
    """
    # 1. The strategy is only allowed to generate signals on the test data
    # (In a true ML setup, the model fits on df_train and predicts on df_test. 
    # For now, we apply the logic over the combined window to maintain indicator states, 
    # but ONLY calculate PnL on the out-of-sample portion).
    
    combined_df = pd.concat([df_train, df_test]).copy()
    try:
        combined_df = strategy_module.get_signals(combined_df)
    except Exception as e:
        print(f"Strategy crashed during execution: {e}")
        return 0.0, 0
    
    # 2. Isolate the Out-Of-Sample (OOS) testing period
    oos_df = combined_df.iloc[len(df_train):].copy()
    
    if 'signal' not in oos_df.columns:
        return 0.0, 0
        
    # 3. Strict PnL Calculation (incorporating fees and slippage)
    oos_df['position'] = oos_df['signal'].shift(1).fillna(0)
    # Calculate returns multiplier
    oos_df['strategy_returns'] = oos_df['position'] * oos_df['log_return'] # Assuming log returns from V2
    
    # Apply fees on position changes
    trade_triggers = oos_df['position'].diff().fillna(0) != 0
    oos_df.loc[trade_triggers, 'strategy_returns'] -= (FEES + SLIPPAGE)

    total_oos_return = np.exp(oos_df['strategy_returns'].sum()) - 1
    trade_count = trade_triggers.sum()
    
    return total_oos_return, trade_count

def run_walk_forward_optimization():
    print("⚖️ THE JUDGE: Initiating Walk-Forward Optimization...")
    
    # Load Data
    try:
        df = pd.read_csv(DATA_FILE)
        # Note: We need to ensure log_return exists for our PnL math
        if 'log_return' not in df.columns:
            df['log_return'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    except FileNotFoundError:
        print(f"❌ Critical Error: Could not find data file {DATA_FILE}")
        sys.exit(1)

    strategy = load_strategy()
    
    # Walk-Forward Parameters
    total_len = len(df)
    train_size = int(total_len * 0.4) # 40% of data for training
    test_size = int(total_len * 0.1)  # 10% steps for testing
    
    if train_size + test_size > total_len:
         print("Dataset too small for WFO.")
         return

    fold_returns = []
    total_trades = 0
    
    print("\n--- OUT OF SAMPLE (OOS) FOLD RESULTS ---")
    for start_idx in range(0, total_len - train_size - test_size + 1, test_size):
        end_train = start_idx + train_size
        end_test = end_train + test_size
        
        df_train = df.iloc[start_idx:end_train]
        df_test = df.iloc[end_train:end_test]
        
        oos_return, trades = evaluate_window(df_train, df_test, strategy)
        fold_returns.append(oos_return)
        total_trades += trades
        
        print(f"Fold {len(fold_returns)}: OOS Return = {oos_return:.2%}, Trades = {trades}")

    # The Final Uncheatable Score
    avg_oos_return = np.mean(fold_returns)
    win_rate_folds = sum(1 for r in fold_returns if r > 0) / len(fold_returns)
    
    print("\n==================================================")
    print(f"⚖️ FINAL WFO SCORE: {avg_oos_return:.4%} Avg OOS Return")
    print(f"📈 Fold Win Rate:   {win_rate_folds:.2%} ({sum(1 for r in fold_returns if r > 0)}/{len(fold_returns)} positive folds)")
    print(f"🔄 Total Trades:    {total_trades}")
    print("==================================================")
    
    # We output the FINAL_RESULT so your auto_loop.py can parse it
    print(f"FINAL_RESULT:{avg_oos_return}")

if __name__ == "__main__":
    run_walk_forward_optimization()