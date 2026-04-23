"""
The Multi-Regime Walk-Forward Judge.
Evaluates strategy.py using strict Out-Of-Sample validation across 
Recent (3M), Medium (1Y), and Macro (3Y) regimes to prevent overfitting.
"""

import pandas as pd
import numpy as np
import importlib.util
import sys
import os

# Configuration
# DATA_FILE = "../data/btc_1h_3y.csv"
DATA_FILE = "../data/btc_1h_3y_v2.csv"
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
    Generates signals, but ONLY evaluates PnL on the strictly unseen test window.
    """
    combined_df = pd.concat([df_train, df_test]).copy()
    try:
        combined_df = strategy_module.get_signals(combined_df)
    except Exception as e:
        print(f"Strategy crashed during execution: {e}")
        return 0.0, 0
    
    # Isolate the Out-Of-Sample (OOS) testing period
    oos_df = combined_df.iloc[len(df_train):].copy()
    
    if 'signal' not in oos_df.columns:
        return 0.0, 0
        
    # Strict PnL Calculation (incorporating fees and slippage)
    oos_df['position'] = oos_df['signal'].shift(1).fillna(0)
    
    # Calculate returns multiplier
    if 'log_return' not in oos_df.columns:
        oos_df['log_return'] = np.log(oos_df['close'] / oos_df['close'].shift(1)).fillna(0)
        
    oos_df['strategy_returns'] = oos_df['position'] * oos_df['log_return']
    
    # Apply fees on position changes
    trade_triggers = oos_df['position'].diff().fillna(0) != 0
    oos_df.loc[trade_triggers, 'strategy_returns'] -= (FEES + SLIPPAGE)

    total_oos_return = np.exp(oos_df['strategy_returns'].sum()) - 1
    trade_count = trade_triggers.sum()
    
    return total_oos_return, trade_count

def evaluate_regime_wfo(regime_df, regime_name, strategy):
    """Runs the Walk-Forward Optimization strictly on a specific regime slice."""
    total_len = len(regime_df)
    train_size = int(total_len * 0.4) # 40% of this regime for training
    test_size = int(total_len * 0.1)  # 10% steps for testing
    
    if train_size + test_size > total_len:
         print(f"⚠️ Dataset too small for WFO in {regime_name}.")
         return -999.0, 0

    fold_returns = []
    total_trades = 0
    
    for start_idx in range(0, total_len - train_size - test_size + 1, test_size):
        end_train = start_idx + train_size
        end_test = end_train + test_size
        
        df_train = regime_df.iloc[start_idx:end_train]
        df_test = regime_df.iloc[end_train:end_test]
        
        oos_return, trades = evaluate_window(df_train, df_test, strategy)
        fold_returns.append(oos_return)
        total_trades += trades

    avg_oos_return = np.mean(fold_returns)
    
    # --- THE LAZY TAX ---
    MINIMUM_TRADES_REQUIRED = 5
    
    if total_trades < MINIMUM_TRADES_REQUIRED:
        print(f"❌ {regime_name}: Lazy Tax Applied. Only took {total_trades} trades across all folds.")
        return -999.0, total_trades

    print(f"✅ {regime_name}: {total_trades} trades | Avg OOS Return: {avg_oos_return:.2%}")
    return avg_oos_return, total_trades

def run_walk_forward_optimization():
    print("⚖️ THE JUDGE: Initiating Walk-Forward Optimization...")
    
    # 1. Load Data
    try:
        df = pd.read_csv(DATA_FILE)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
        if 'log_return' not in df.columns:
            df['log_return'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    except FileNotFoundError:
        print(f"❌ Critical Error: Could not find data file {DATA_FILE}")
        sys.exit(1)

    # 2. Load the Agent's Strategy
    try:
        strategy = load_strategy()
    except Exception as e:
        print(f"⚠️ Strategy compilation crashed: {e}")
        print("FINAL_RESULT:-999.0")
        return

    # 3. PURE SLIDING WINDOW EVALUATION
    # evaluate_regime_wfo already handles training on 40% and testing on 10% steps!
    print("\n--- 3-YEAR CONTINUOUS WFO EVALUATION ---")
    final_score, total_trades = evaluate_regime_wfo(df, "3-Year Macro Pipeline", strategy)

    print("\n==================================================")
    # Multiply by 100 to feed the AI a clean Percentage Score (e.g., -27.08 instead of -0.27)
    percentage_score = final_score * 100 if final_score != -999.0 else -999.0
    print(f"⚖️ FINAL SCORE (Avg OOS Return per fold): {percentage_score:.4f}%")
    print("==================================================")
    
    # Output strictly for auto_loop.py regex parsing
    print(f"FINAL_RESULT:{percentage_score}")

if __name__ == "__main__":
    run_walk_forward_optimization()