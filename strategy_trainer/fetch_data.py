"""
fetch_data.py
A data pipeline to fetch and stitch 3 years of historical OHLCV data.
Run this before starting auto_loop.py if your data is missing.
"""

import os
import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta, timezone

# Configuration
SYMBOL = 'BTC/USDT'
TIMEFRAME = '15m'
YEARS_TO_FETCH = 3
LIMIT_PER_REQUEST = 1000

# Pathing
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "btc_15m_3y.csv")

def fetch_historical_data():
    print(f"📡 Initializing connection to Binance via CCXT...")
    exchange = ccxt.binance({
        'enableRateLimit': True, # Crucial to prevent IP bans
    })

    # Ensure the data directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if os.path.exists(OUTPUT_FILE):
        print(f"✅ Data already exists at {OUTPUT_FILE}. Skipping download.")
        return

    # Calculate timestamps
    now = datetime.now(timezone.utc)
    start_time = now - timedelta(days=365 * YEARS_TO_FETCH)
    since = int(start_time.timestamp() * 1000) # CCXT requires milliseconds
    end_time = int(now.timestamp() * 1000)

    print(f"🗓️ Fetching {YEARS_TO_FETCH} years of {TIMEFRAME} data for {SYMBOL}...")
    print(f"⏳ Start Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    all_candles = []
    
    while since < end_time:
        try:
            # Fetch the batch
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since, LIMIT_PER_REQUEST)
            
            if not ohlcv:
                break # We've reached the end of the available data
                
            all_candles.extend(ohlcv)
            
            # Update the 'since' variable to the timestamp of the last candle + 1 tick
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 1
            
            # Print progress
            readable_date = datetime.fromtimestamp(last_timestamp / 1000, tz=timezone.utc).strftime('%Y-%m-%d')
            print(f"   [+] Fetched data up to: {readable_date} | Total Candles: {len(all_candles)}")
            
            time.sleep(0.5) # Be gentle to the exchange API

        except Exception as e:
            print(f"⚠️ Exchange connection error: {e}")
            print("Retrying in 10 seconds...")
            time.sleep(10)

    print("\n🔨 Processing and formatting data...")
    # Convert to Pandas DataFrame
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Format the timestamp exactly how prepare.py expects it
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ SUCCESS! 3-Year dataset saved to: {OUTPUT_FILE}")
    print(f"📊 Total Rows: {len(df)}")

if __name__ == "__main__":
    fetch_historical_data()