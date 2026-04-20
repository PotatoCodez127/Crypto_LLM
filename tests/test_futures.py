import sys
import os

# This tells Python to look one folder UP from the 'tests' folder,
# automatically finding your main Crypto_LLM folder and the 'src' directory.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.data_feed.handler import DataHandler

if __name__ == "__main__":
    handler = DataHandler()
    
    # Try to pull the last 5 hours of Futures data + OI
    df = handler.get_full_market_data(symbol="BTC/USDT:USDT", timeframe="1h", limit=5)
    
    print("\n✅ LIVE FUTURES DATA:")
    print(df[['timestamp', 'close', 'open_interest']])