import pandas as pd
import numpy as np
from strategy import get_signals

def evaluate_strategy(timeframe='1y'):
    """
    V2 Judge: Walk-Forward Optimization (Train/Test Split)
    """
    csv_path = f"../data/btc_1h_{timeframe}.csv"
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # THE SPLIT (60% Train / 40% Test)
    split_idx = int(len(df) * 0.6)
    df_train = df.iloc[:split_idx].copy().reset_index(drop=True)
    df_test = df.iloc[split_idx:].copy().reset_index(drop=True)

    def score_dataset(data):
        evaluated_df = get_signals(data)
        
        evaluated_df['position'] = evaluated_df['signal'].shift(1).fillna(0)
        evaluated_df['trade_return'] = evaluated_df['position'] * evaluated_df['log_ret']
        
        evaluated_df['cum_return'] = evaluated_df['trade_return'].cumsum()
        
        total_return = np.exp(evaluated_df['cum_return'].iloc[-1]) - 1 if len(evaluated_df) > 0 else 0
        
        mean_ret = evaluated_df['trade_return'].mean()
        std_ret = evaluated_df['trade_return'].std()
        sharpe = (mean_ret / std_ret) * np.sqrt(8760) if std_ret > 0 else 0
        
        roll_max = evaluated_df['cum_return'].cummax()
        drawdown = evaluated_df['cum_return'] - roll_max
        max_dd = np.exp(drawdown.min()) - 1 if len(drawdown) > 0 else 0

        raw_score = (sharpe * 0.40) + (total_return * 0.40) + ((1 + max_dd) * 0.20)
        
        trade_count = (evaluated_df['signal'].diff() != 0).sum()
        if trade_count < 30:
            raw_score *= (trade_count / 30)
            
        return raw_score, total_return, sharpe, max_dd, trade_count

    # Score Both Halves Independently
    train_score, train_ret, train_sh, train_dd, train_tc = score_dataset(df_train)
    test_score, test_ret, test_sh, test_dd, test_tc = score_dataset(df_test)

    # The final score penalizes variance between the train and test sets
    variance_penalty = abs(train_score - test_score) * 0.2
    final_score = min(train_score, test_score) - variance_penalty

    print("\n--- WALK-FORWARD EVALUATION REPORT ---")
    print(f"TRAIN SET (First 60%):")
    print(f"  Return: {train_ret*100:.2f}% | Sharpe: {train_sh:.2f} | Max DD: {train_dd*100:.2f}% | Trades: {train_tc}")
    print(f"  Train Score: {train_score:.4f}")
    
    print(f"\nTEST SET (Last 40%):")
    print(f"  Return: {test_ret*100:.2f}% | Sharpe: {test_sh:.2f} | Max DD: {test_dd*100:.2f}% | Trades: {test_tc}")
    print(f"  Test Score: {test_score:.4f}")
    
    print(f"\nVariance Penalty: -{variance_penalty:.4f}")
    print(f"FINAL_RESULT:{final_score}")

    return final_score