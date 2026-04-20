import os
import pandas as pd
import ccxt
import numpy as np
import time
import argparse
from tqdm import tqdm

# ---------------- CONFIGURATION ----------------
SYMBOL = 'BTC/USDT:USDT'
TIMEFRAME = '1h'

TIMEFRAME_MULTIPLIERS = {
    '1m': (60 * 24 * 365),
    '5m': (12 * 24 * 365),
    '15m': (4 * 24 * 365),
    '1h': (24 * 365),
    '4h': (6 * 365),
    '1d': 365
}

PERIOD_MAP = {
    '3y': 3 * 365,
    '1y': 365,
    '3m': 90
}

FEE = 0.0015  
MIN_TRADES = 50 

# ---------------- DATA ENGINE ----------------
def fetch_data(target_period):
    """
    Forces the data folder to live at the root: C:/Users/Stefa/OneDrive/Desktop/Crypto_LLM/data
    """
    # Use .replace('\\', '/') to keep Windows paths from breaking the string
    script_dir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
    
    # Move up one level to 'Crypto_LLM'
    root_dir = os.path.dirname(script_dir) 
    data_dir = os.path.join(root_dir, "data").replace('\\', '/')
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"[*] Created global directory: {data_dir}")

    clean_tf = TIMEFRAME.replace('/', '').replace(':', '')
    filename = os.path.join(data_dir, f"btc_{clean_tf}_{target_period}.csv").replace('\\', '/')
    
    exchange = ccxt.bybit({'enableRateLimit': True})

    # 2. Handle Local Storage
    if os.path.exists(filename):
        # We use index_col=0 to ensure the timestamp is the ID
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        # Force index to be datetime objects
        df.index = pd.to_datetime(df.index)
        print(f"[*] Found local data: {len(df)} candles in {filename}")
    else:
        df = pd.DataFrame()

    # 3. Determine Start Point & Calculate Total Needed
    # FIX: Ensure we use the full period for fresh downloads
    days_to_fetch = PERIOD_MAP.get(target_period, 90)
    now_ms = exchange.milliseconds()
    
    if not df.empty:
        # Resume from the last millisecond we have
        last_ts = int(df.index[-1].timestamp() * 1000)
        since = last_ts + 1
        ms_diff = now_ms - last_ts
        # Estimate missing hours
        total_needed = max(1, int(ms_diff / (1000 * 60 * 60)))
        print(f"[*] Resuming download for {target_period}...")
    else:
        # Start from the full lookback period
        since = now_ms - (days_to_fetch * 24 * 60 * 60 * 1000)
        total_needed = days_to_fetch * 24
        print(f"[*] Fresh start: Fetching {days_to_fetch} days for {target_period}...")

    # 4. Pagination Loop
    all_ohlcv = []
    batch_size = 200
    pbar = tqdm(total=total_needed, desc=f"Progress ({target_period})")
    
    # Use a safety counter to prevent infinite loops
    max_retries = 5
    retry_count = 0

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, since=since, limit=batch_size)
            
            if not ohlcv or len(ohlcv) == 0:
                # If we get nothing, try one more time before giving up
                retry_count += 1
                if retry_count > max_retries:
                    break
                time.sleep(2)
                continue

            retry_count = 0 # Reset retries on success
            all_ohlcv.extend(ohlcv)
            
            # Move 'since' forward to the last candle received + 1ms
            last_candle_ts = ohlcv[-1][0]
            since = last_candle_ts + 1 
            
            pbar.update(len(ohlcv))

            # NEW LOGIC: Stop ONLY if the last candle is within the last hour
            # This ensures we don't stop early just because a batch was small
            if last_candle_ts >= (exchange.milliseconds() - 3600000):
                print("\n[*] Reached the current market time.")
                break

            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            pbar.write(f"[!] API Error: {e}")
            time.sleep(5)
            continue
            
    pbar.close()

    # 5. Merge, Clean, and Save
    if all_ohlcv:
        new_df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
        new_df.set_index('timestamp', inplace=True)
        
        # Combine old data with new
        df = pd.concat([df, new_df])
        # Drop any overlapping candles (keeps the most recent version)
        df = df[~df.index.duplicated(keep='last')]
        df.sort_index(inplace=True)
        
        # Save to the specific path
        df.to_csv(filename)
        print(f"[*] {filename} updated. Total rows: {len(df)}")

    return df

# ---------------- STRATEGY JUDGE ----------------
def evaluate_strategy(timeframe='1y'):
    """
    V2 Judge: Walk-Forward Optimization (Train/Test Split)
    """
    from strategy import get_signals
    import pandas as pd
    import numpy as np

    # 1. Load your data (assuming this part of your code stays the same)
    csv_path = f"../data/btc_1h_{timeframe}.csv"
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 2. THE SPLIT (60% Train / 40% Test)
    split_idx = int(len(df) * 0.6)
    df_train = df.iloc[:split_idx].copy().reset_index(drop=True)
    df_test = df.iloc[split_idx:].copy().reset_index(drop=True)

    # Helper function to score a specific dataset
    def score_dataset(data):
        # Pass the data to the AI's strategy
        evaluated_df = get_signals(data)
        
        # Calculate PnL (assuming 1 = long, -1 = short, 0 = flat)
        # Shift signals by 1 so we enter on the NEXT candle's open
        evaluated_df['position'] = evaluated_df['signal'].shift(1).fillna(0)
        evaluated_df['trade_return'] = evaluated_df['position'] * evaluated_df['log_ret'] # using log returns from strategy.py
        
        # Cumulative returns
        evaluated_df['cum_return'] = evaluated_df['trade_return'].cumsum()
        
        # Calculate Metrics
        total_return = np.exp(evaluated_df['cum_return'].iloc[-1]) - 1 if len(evaluated_df) > 0 else 0
        
        # Sharpe Ratio (Annualized for 1h candles: 24 * 365 = 8760)
        mean_ret = evaluated_df['trade_return'].mean()
        std_ret = evaluated_df['trade_return'].std()
        sharpe = (mean_ret / std_ret) * np.sqrt(8760) if std_ret > 0 else 0
        
        # Drawdown
        roll_max = evaluated_df['cum_return'].cummax()
        drawdown = evaluated_df['cum_return'] - roll_max
        max_dd = np.exp(drawdown.min()) - 1 if len(drawdown) > 0 else 0

        # Simple Scoring Formula
        raw_score = (sharpe * 0.40) + (total_return * 0.40) + ((1 + max_dd) * 0.20)
        
        # Trade Count Penalty (Must take at least 30 trades per period)
        trade_count = (evaluated_df['signal'].diff() != 0).sum()
        if trade_count < 30:
            raw_score *= (trade_count / 30)
            
        return raw_score, total_return, sharpe, max_dd, trade_count

    # 3. Score Both Halves Independently
    train_score, train_ret, train_sh, train_dd, train_tc = score_dataset(df_train)
    test_score, test_ret, test_sh, test_dd, test_tc = score_dataset(df_test)

    # 4. THE UNCHEATABLE METRIC
    # The final score is heavily penalized if the model overfits the training data
    # We take the minimum of the two scores, minus a penalty for high variance between them.
    variance_penalty = abs(train_score - test_score) * 0.2
    final_score = min(train_score, test_score) - variance_penalty

    # 5. Output the Detailed Report (Captured by auto_loop.py)
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
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--period', type=str, default='1y', choices=['3m', '1y', '3y'])
    args = parser.parse_args()

    final_score = evaluate_strategy(args.period)
    print(f"FINAL_RESULT:{final_score}")