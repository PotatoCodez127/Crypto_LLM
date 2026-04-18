import os
import sys
import pandas as pd
import ccxt
import numpy as np
import argparse

# --- CONFIGURATION ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'  # Change to '15m' easily here
DATA_FILE = "btc_data.csv"

def fetch_data():
    """Downloads historical data if it doesn't exist."""
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    
    print(f"Downloading {SYMBOL} {TIMEFRAME} data...")
    exchange = ccxt.binance()
    # Fetch 2000 candles (roughly 3 months of 1H data)
    ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=1000)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.to_csv(DATA_FILE)
    return df

def evaluate_strategy():
    """
    This is the 'Judge'. It imports the AI's strategy and scores it.
    This replaces the 'evaluate_bpb' function from the original code.
    """
    try:
        # Dynamically import the strategy the AI is working on
        from strategy import get_signals
        df = fetch_data()
        
        # Get signals from the AI's logic
        df = get_signals(df)
        
        # Calculate Returns
        df['market_log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['strat_ret'] = df['signal'].shift(1) * df['market_log_ret']
        
        # Apply slippage/fees (0.05% per trade)
        trades = df['signal'].diff().fillna(0).abs()
        df['strat_ret'] -= trades * 0.0005
        
        # Calculate Sharpe Ratio
        # Annualizing factor: 1H -> sqrt(24*365), 15m -> sqrt(4*24*365)
        annual_factor = np.sqrt(24 * 365) if TIMEFRAME == '1h' else np.sqrt(96 * 365)
        
        mean_ret = df['strat_ret'].mean()
        std_ret = df['strat_ret'].std()
        
        if std_ret == 0 or np.isnan(std_ret):
            return 0.0
            
        sharpe = (mean_ret / std_ret) * annual_factor
        return sharpe
    except Exception as e:
        print(f"Error evaluating strategy: {e}")
        return -1.0

if __name__ == "__main__":
    # This main block allows the Research AI to run 'python prepare.py' 
    # to get a single number representing performance.
    score = evaluate_strategy()
    print(f"SCORE: {score}")