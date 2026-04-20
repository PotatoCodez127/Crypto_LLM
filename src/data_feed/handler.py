import ccxt
import pandas as pd
from src.config.settings import EXCHANGE_ID

class DataHandler:
    def __init__(self):
        # Connect to the Exchange ANONYMOUSLY (No API Keys needed for public data)
        exchange_class = getattr(ccxt, EXCHANGE_ID)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future', # Force ccxt to use the Perpetual Futures market
            }
        })
        
    def fetch_ohlcv(self, symbol="BTC/USDT:USDT", timeframe="1h", limit=1000):
        print(f"Fetching {limit} candles of {symbol} {timeframe} Futures data...")
        bars = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def fetch_open_interest(self, symbol="BTC/USDT:USDT", timeframe="1h", limit=1000):
        try:
            print(f"Fetching Open Interest for {symbol}...")
            oi_data = self.exchange.fetch_open_interest_history(symbol, timeframe, limit=limit)
            
            clean_oi = []
            for entry in oi_data:
                clean_oi.append({
                    'timestamp': pd.to_datetime(entry['timestamp'], unit='ms'),
                    'open_interest': entry['openInterestValue']
                })
            
            return pd.DataFrame(clean_oi)
        
        except Exception as e:
            print(f"⚠️ Exchange {EXCHANGE_ID} might not support historical OI directly via ccxt: {e}")
            return pd.DataFrame(columns=['timestamp', 'open_interest'])

    def get_full_market_data(self, symbol="BTC/USDT:USDT", timeframe="1h", limit=1000):
        """Merges Price Data with Open Interest into one master dataframe."""
        df_price = self.fetch_ohlcv(symbol, timeframe, limit)
        df_oi = self.fetch_open_interest(symbol, timeframe, limit)
        
        if not df_oi.empty:
            df_merged = pd.merge(df_price, df_oi, on='timestamp', how='left')
            df_merged['open_interest'] = df_merged['open_interest'].ffill().fillna(0)
            return df_merged
        else:
            df_price['open_interest'] = 0.0
            return df_price