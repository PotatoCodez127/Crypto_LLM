"""
fetch_data.py
A data pipeline to fetch and stitch historical OHLCV data.
Automatically detects existing datasets and appends only the missing recent candles.
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
LIMIT_PER_REQUEST = 1000  # Binance max

# Pathing (Resolves to the /data folder in your root repo)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
# Dynamically name the file based on the selected timeframe
OUTPUT_FILE = os.path.join(DATA_DIR, f"btc_{TIMEFRAME}_3y.csv")

def fetch_historical_data():
    print(f"📡 Initializing connection to Binance via CCXT...")
    exchange = ccxt.binance({
        'enableRateLimit': True, # Crucial to prevent IP bans
    })

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    now = datetime.now(timezone.utc)
    end_time = int(now.timestamp() * 1000)
    
    # Check for existing data to append missing candles instead of starting over
    if os.path.exists(OUTPUT_FILE):
        print(f"🔍 Existing data found at {OUTPUT_FILE}. Checking for missing candles...")
        existing_df = pd.read_csv(OUTPUT_FILE)
        existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
        
        # Get the very last timestamp in the file, convert to milliseconds
        last_dt = existing_df['timestamp'].max()
        since = int(last_dt.timestamp() * 1000) + 1
        
        print(f"🗓️ Last recorded candle: {last_dt}. Fetching data from here to present...")
        append_mode = True
    else:
        print(f"🗓️ No existing data. Fetching {YEARS_TO_FETCH} years of {TIMEFRAME} data for {SYMBOL}...")
        start_time = now - timedelta(days=365 * YEARS_TO_FETCH)
        since = int(start_time.timestamp() * 1000)
        append_mode = False

    # Prevent fetching if the data is already up to the current minute
    if since >= end_time:
        print("✅ Data is already completely up to date. No missing candles to fetch.")
        return

    all_candles = []
    
    while since < end_time:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since, LIMIT_PER_REQUEST)
            
            if not ohlcv:
                break 
                
            all_candles.extend(ohlcv)
            
            # Update the 'since' variable to the timestamp of the last candle + 1 tick
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 1
            
            readable_date = datetime.fromtimestamp(last_timestamp / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
            print(f"   [+] Fetched missing data up to: {readable_date} | New Candles: {len(all_candles)}")
            
            time.sleep(0.5) 

        except Exception as e:
            print(f"⚠️ Exchange connection error: {e}")
            print("Retrying in 10 seconds...")
            time.sleep(10)

    if not all_candles:
        print("✅ Data is completely up to date.")
        return

    print("\n🔨 Processing and formatting new data...")
    new_df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
    
    if append_mode:
        # Append to existing and drop any accidental overlap duplicates at the merge point
        combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        combined_df.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ SUCCESS! {len(new_df)} missing candles appended to {OUTPUT_FILE}")
    else:
        new_df.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ SUCCESS! 3-Year dataset saved to: {OUTPUT_FILE}")
        
    print(f"📊 Total Rows Now: {len(combined_df) if append_mode else len(new_df)}")

if __name__ == "__main__":
    fetch_historical_data()