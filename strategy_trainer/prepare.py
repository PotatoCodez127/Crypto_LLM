import pandas as pd
import numpy as np
import importlib.util
import sys
import os

# Clean, foolproof pathing: Jump up exactly one level to the Crypto_LLM root directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

DATA_FILE = os.path.join(PROJECT_ROOT, "data", "btc_1h_3y.csv")
STRATEGY_FILE = "strategy.py"
FEES = 0.0006  
SLIPPAGE = 0.0005

def load_strategy():
    spec = importlib.util.spec_from_file_location("strategy", STRATEGY_FILE)
    strategy = importlib.util.module_from_spec(spec)
    sys.modules["strategy"] = strategy
    spec.loader.exec_module(strategy)
    return strategy

def calc_metrics(df):
    """Calculates all institutional metrics for a given train or test dataframe"""
    if 'signal' not in df.columns or len(df) == 0:
        return 0, 0, 0, 0, 0, 0

    # FIX: Reset the index to prevent Pandas .loc collisions from overlapping WFO folds
    df = df.reset_index(drop=True).copy()
    
    df['position'] = df['signal'].shift(1).fillna(0)
    
    if 'log_return' not in df.columns:
        df['log_return'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        
    df['strategy_returns'] = df['position'] * df['log_return']
    trade_triggers = df['position'].diff().fillna(0) != 0
    df.loc[trade_triggers, 'strategy_returns'] -= (FEES + SLIPPAGE)

    trade_count = trade_triggers.sum()
    if trade_count == 0:
        return 0, 0, 0, 0, 0, 0

    total_return = (np.exp(df['strategy_returns'].sum()) - 1) * 100

    cum_exp = np.exp(df['strategy_returns'].cumsum())
    max_dd = ((cum_exp / cum_exp.cummax()) - 1).min() * 100

    mean_r = df['strategy_returns'].mean()
    std_r = df['strategy_returns'].std()
    sharpe = (mean_r / std_r) * np.sqrt(8760) if std_r > 0 else 0

    df['trade_id'] = trade_triggers.cumsum()
    trades_df = df[df['position'] != 0].groupby('trade_id')['strategy_returns'].sum()
    wins = trades_df[trades_df > 0]
    losses = trades_df[trades_df <= 0]
    
    win_rate = (len(wins) / len(trades_df)) * 100 if len(trades_df) > 0 else 0
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else 99.99

    return total_return, win_rate, pf, max_dd, sharpe, trade_count

def evaluate_window(df_train, df_test, strategy_module):
    combined_df = pd.concat([df_train, df_test]).copy()
    try:
        combined_df = strategy_module.get_signals(combined_df)
    except Exception as e:
        print(f"Strategy crashed: {e}")
        return None, None
    
    # Isolate Train and Test datasets based on the original lengths
    ins_df = combined_df.iloc[:len(df_train)].copy()
    oos_df = combined_df.iloc[len(df_train):].copy()
    
    return ins_df, oos_df

def evaluate_regime_wfo(regime_df, strategy):
    total_len = len(regime_df)
    train_size = int(total_len * 0.4) 
    test_size = int(total_len * 0.1)  
    
    if train_size + test_size > total_len:
         return None, None

    master_ins_list = []
    master_oos_list = []
    
    for start_idx in range(0, total_len - train_size - test_size + 1, test_size):
        end_train = start_idx + train_size
        end_test = end_train + test_size
        
        df_train = regime_df.iloc[start_idx:end_train]
        df_test = regime_df.iloc[end_train:end_test]
        
        ins_df, oos_df = evaluate_window(df_train, df_test, strategy)
        if ins_df is not None and oos_df is not None:
            master_ins_list.append(ins_df)
            master_oos_list.append(oos_df)

    if not master_oos_list:
        return None, None

    # Concatenate all folds to calculate macro statistics
    master_ins = pd.concat(master_ins_list)
    master_oos = pd.concat(master_oos_list)
    
    ins_metrics = calc_metrics(master_ins)
    oos_metrics = calc_metrics(master_oos)
    
    return ins_metrics, oos_metrics

def print_row(regime, phase, m):
    if m is None or m[5] == 0:
        print(f"| {regime:<7} | {phase:<5} | {'N/A':<6} | {'N/A':<8} | {'N/A':<8} | {'N/A':<6} | {'N/A':<8} | {'N/A':<6} |")
    else:
        ret, wr, pf, dd, sh, tr = m
        print(f"| {regime:<7} | {phase:<5} | {tr:<6} | {ret:>7.2f}% | {wr:>7.2f}% | {pf:>6.2f} | {dd:>7.2f}% | {sh:>6.2f} |")

def run_walk_forward_optimization():
    print("⚖️ THE JUDGE: Initiating Multi-Timeframe Walk-Forward Optimization...")
    
    try:
        df = pd.read_csv(DATA_FILE)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
        if 'log_return' not in df.columns:
            df['log_return'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    except FileNotFoundError:
        # Fallback for alternative dataset name
        df = pd.read_csv(DATA_FILE.replace("btc_1h_3y.csv", "btc_1h_3y_v2.csv"))

    strategy = load_strategy()

    # Define Horizons (1 hour candles)
    df_3y = df
    df_1y = df.tail(8760)
    df_3m = df.tail(2160)

    # Evaluate Horizons
    ins_3y, oos_3y = evaluate_regime_wfo(df_3y, strategy)
    ins_1y, oos_1y = evaluate_regime_wfo(df_1y, strategy)
    ins_3m, oos_3m = evaluate_regime_wfo(df_3m, strategy)

    print("\n" + "="*80)
    print("📊 INSTITUTIONAL STRATEGY TEARSHEET (TRAIN vs TEST)")
    print("="*80)
    print("| Horizon | Phase | Trades | Return   | Win Rate | Profit | Max DD   | Sharpe |")
    print("|---------|-------|--------|----------|----------|--------|----------|--------|")
    
    print_row("3-Year", "Train", ins_3y)
    print_row("3-Year", "Test", oos_3y)
    print("|---------|-------|--------|----------|----------|--------|----------|--------|")
    print_row("1-Year", "Train", ins_1y)
    print_row("1-Year", "Test", oos_1y)
    print("|---------|-------|--------|----------|----------|--------|----------|--------|")
    print_row("3-Month", "Train", ins_3m)
    print_row("3-Month", "Test", oos_3m)
    print("="*80)

    # The AI is judged strictly on the 3-Year Out-Of-Sample Returns
    if oos_3y and oos_3y[5] >= 5: 
        final_score = oos_3y[0]
    else:
        final_score = -999.0
        print("❌ Lazy Tax Applied: Insufficient 3-Year OOS Trades.")
        
    print(f"\nFINAL_RESULT:{final_score}")

if __name__ == "__main__":
    run_walk_forward_optimization()