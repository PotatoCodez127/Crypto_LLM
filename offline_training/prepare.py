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
def evaluate_strategy(target_period):
    try:
        import importlib
        import strategy
        importlib.reload(strategy)
        from strategy import get_signals

        # 1. Get Data
        df = fetch_data(target_period)
        if df.empty: return -1.0

        # 2. Run Strategy
        df = get_signals(df)
        if 'signal' not in df.columns: return -1.0
        df['signal'] = np.sign(df['signal'].fillna(0))
        
        # 3. Returns & Fees
        df['market_log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['strat_ret'] = df['signal'].shift(1) * df['market_log_ret']
        trades_series = df['signal'].diff().abs().fillna(0)
        df['strat_ret'] -= (trades_series * FEE)
        df = df.dropna()

        # 4. Metrics Calculation
        df['equity'] = np.exp(df['strat_ret'].cumsum())
        df['peak'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] - df['peak']) / df['peak']
        
        max_dd = df['drawdown'].min()
        total_return = df['equity'].iloc[-1] - 1
        returns = df['strat_ret']
        
        # Sharpe Calculation
        ann_factor = np.sqrt(TIMEFRAME_MULTIPLIERS.get(TIMEFRAME, 8760))
        std = returns.std()
        sharpe = (returns.mean() / std) * ann_factor if std > 1e-8 else -1.0
        
        # Trade Analysis
        trade_count = int(trades_series.sum() / 2)
        win_rate = (returns > 0).mean() if trade_count > 0 else 0
        
        # RR Calculation (Avg Win / Avg Loss)
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 1e-8
        rr_ratio = avg_win / avg_loss

        # --- SENIOR LOGIC: EXPECTANCY & CONSISTENCY ---
        # Expectancy: (Win Rate * Avg Win) - (Loss Rate * Avg Loss)
        # Tells us if the strategy is mathematically sound after fees
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # Yearly Consistency: Check for any years where the strategy lost money
        yearly_rets = df.groupby(df.index.year)['strat_ret'].sum()
        losing_years = int((yearly_rets < 0).sum())

        # 5. FIXED MATH: Raw Score (Including Expectancy)
        # We increase the weight of Sharpe and Expectancy to prioritize quality over raw luck
        raw_score = (sharpe * 0.3) + (total_return * 0.2) + (expectancy * 100 * 0.3) + ((1 + max_dd) * 0.2)

        # 6. REWARDS vs PENALTIES Logic (Additive)
        adjustment = 0.0
        
        # PENALTIES
        if trade_count < MIN_TRADES:
            adjustment -= 2.0  
        if max_dd < -0.25:
            adjustment -= 1.5  
        if total_return < 0:
            adjustment -= 1.0  
        if win_rate < 0.40:
            adjustment -= 0.5  
        if expectancy < FEE:
            adjustment -= 1.0 # Strategy is being eaten alive by fees
        if losing_years > 0:
            adjustment -= (losing_years * 0.5) # Penalty for every losing year

        # REWARDS
        if max_dd > -0.15 and trade_count >= MIN_TRADES:
            adjustment += 0.5  
        if win_rate > 0.60:
            adjustment += 0.5  
        if total_return > 0.50:
            adjustment += 1.0  
        if expectancy > (FEE * 3):
            adjustment += 1.0 # High quality "Alpha"

        final_score = raw_score + adjustment

        print(f"""
            --- PERFORMANCE REPORT [{target_period}] ---
            Sharpe Ratio  : {sharpe:.2f}
            Total Return  : {total_return:.2%}
            Max Drawdown  : {max_dd:.2%}
            Win Rate      : {win_rate:.2%}
            Risk:Reward   : {rr_ratio:.2f}
            Expectancy    : {expectancy:.4f}
            Losing Years  : {losing_years}
            Total Trades  : {trade_count}
            ---------------------------------
            RAW SCORE     : {raw_score:.4f}
            ADJUSTMENT    : {adjustment:+.2f}
            FINAL SCORE   : {final_score:.4f}
            """)
            
        return final_score

    except Exception as e:
        print(f"[!] Critical Judge Error: {e}")
        import traceback
        traceback.print_exc()
        return -10.0
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--period', type=str, default='1y', choices=['3m', '1y', '3y'])
    args = parser.parse_args()

    final_score = evaluate_strategy(args.period)
    print(f"FINAL_RESULT:{final_score}")