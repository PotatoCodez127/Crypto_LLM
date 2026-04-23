import pandas as pd
import numpy as np
import importlib.util
import sys
import os

DATA_FILE = "../data/btc_1h_3y_v2.csv"
STRATEGY_FILE = "strategy.py"
INITIAL_CAPITAL = 10000.0
FEES = 0.0006  
SLIPPAGE = 0.0005 

def load_strategy():
    spec = importlib.util.spec_from_file_location("strategy", STRATEGY_FILE)
    strategy = importlib.util.module_from_spec(spec)
    sys.modules["strategy"] = strategy
    spec.loader.exec_module(strategy)
    return strategy

def evaluate_window(df_train, df_test, strategy_module):
    combined_df = pd.concat([df_train, df_test]).copy()
    try:
        combined_df = strategy_module.get_signals(combined_df)
    except Exception as e:
        print(f"Strategy crashed during execution: {e}")
        return None, 0
    
    oos_df = combined_df.iloc[len(df_train):].copy()
    
    if 'signal' not in oos_df.columns:
        return None, 0
        
    oos_df['position'] = oos_df['signal'].shift(1).fillna(0)
    
    if 'log_return' not in oos_df.columns:
        oos_df['log_return'] = np.log(oos_df['close'] / oos_df['close'].shift(1)).fillna(0)
        
    oos_df['strategy_returns'] = oos_df['position'] * oos_df['log_return']
    
    trade_triggers = oos_df['position'].diff().fillna(0) != 0
    oos_df.loc[trade_triggers, 'strategy_returns'] -= (FEES + SLIPPAGE)

    trade_count = trade_triggers.sum()
    
    # Return the dataframe instead of just the score so we can calculate advanced metrics
    return oos_df, trade_count

def evaluate_regime_wfo(regime_df, regime_name, strategy):
    total_len = len(regime_df)
    train_size = int(total_len * 0.4) 
    test_size = int(total_len * 0.1)  
    
    if train_size + test_size > total_len:
         print(f"⚠️ Dataset too small for WFO.")
         return -999.0, 0

    fold_returns = []
    total_trades = 0
    master_oos_list = []
    
    for start_idx in range(0, total_len - train_size - test_size + 1, test_size):
        end_train = start_idx + train_size
        end_test = end_train + test_size
        
        df_train = regime_df.iloc[start_idx:end_train]
        df_test = regime_df.iloc[end_train:end_test]
        
        oos_df, trades = evaluate_window(df_train, df_test, strategy)
        if oos_df is None:
            continue
            
        fold_returns.append(np.exp(oos_df['strategy_returns'].sum()) - 1)
        total_trades += trades
        master_oos_list.append(oos_df)

    if not master_oos_list or not fold_returns:
        return -999.0, 0

    avg_oos_return = np.mean(fold_returns)
    
    MINIMUM_TRADES_REQUIRED = 5
    if total_trades < MINIMUM_TRADES_REQUIRED:
        print(f"❌ Lazy Tax Applied. Only took {total_trades} trades.")
        return -999.0, total_trades

    # ==========================================
    # ADVANCED METRICS (From get_results.py)
    # ==========================================
    master_oos = pd.concat(master_oos_list)
    master_oos['cum_returns'] = master_oos['strategy_returns'].cumsum()
    master_oos['cum_returns_exp'] = np.exp(master_oos['cum_returns'])
    
    # Max Drawdown
    rolling_max = master_oos['cum_returns_exp'].cummax()
    drawdown = master_oos['cum_returns_exp'] / rolling_max - 1
    max_dd = drawdown.min() * 100

    # Sharpe Ratio (Assuming 1h timeframe, 8760 periods/yr)
    mean_return = master_oos['strategy_returns'].mean()
    std_return = master_oos['strategy_returns'].std()
    sharpe = (mean_return / std_return) * np.sqrt(8760) if std_return > 0 else 0

    # Win Rate & Profit Factor
    master_oos['trade_id'] = (master_oos['position'].diff() != 0).cumsum()
    trades_df = master_oos[master_oos['position'] != 0].groupby('trade_id')['strategy_returns'].sum()
    
    winning_trades = trades_df[trades_df > 0]
    losing_trades = trades_df[trades_df <= 0]
    
    win_rate = (len(winning_trades) / len(trades_df)) * 100 if len(trades_df) > 0 else 0
    gross_profit = winning_trades.sum()
    gross_loss = abs(losing_trades.sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Print the clean tabular report
    print(f"\n✅ {total_trades} trades executed across all folds.")
    print("="*60)
    print("📊 INSTITUTIONAL METRICS (Out-Of-Sample Aggregated)")
    print("="*60)
    print(f"Avg Fold Return : {avg_oos_return*100:.2f}%")
    print(f"Win Rate        : {win_rate:.2f}%")
    print(f"Profit Factor   : {pf:.2f}")
    print(f"Max Drawdown    : {max_dd:.2f}%")
    print(f"Sharpe Ratio    : {sharpe:.2f}")
    print("="*60)
    
    return avg_oos_return, total_trades

def run_walk_forward_optimization():
    print("⚖️ THE JUDGE: Initiating Walk-Forward Optimization...")
    
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

    try:
        strategy = load_strategy()
    except Exception as e:
        print(f"⚠️ Strategy compilation crashed: {e}")
        print("FINAL_RESULT:-999.0")
        return

    print("\n--- 3-YEAR CONTINUOUS WFO EVALUATION ---")
    final_score, total_trades = evaluate_regime_wfo(df, "3-Year Macro Pipeline", strategy)

    print("\n==================================================")
    percentage_score = final_score * 100 if final_score != -999.0 else -999.0
    print(f"⚖️ FINAL SCORE (Avg OOS Return per fold): {percentage_score:.4f}%")
    print("==================================================")
    
    print(f"FINAL_RESULT:{percentage_score}")

if __name__ == "__main__":
    run_walk_forward_optimization()