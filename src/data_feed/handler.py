import ccxt
import pandas as pd
import time
from src.config.settings import EXCHANGE_ID, API_KEY, API_SECRET

class DataHandler:
    def __init__(self):
        # 1. Connect to the Exchange (Upgraded to Futures)
        exchange_class = getattr(ccxt, EXCHANGE_ID)
        self.exchange = exchange_class({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future', # Force ccxt to use the Perpetual Futures market
            }
        })
        
    def fetch_ohlcv(self, symbol="BTC/USDT:USDT", timeframe="1h", limit=1000):
        """Fetches standard price data for the Perpetual contract."""
        print(f"Fetching {limit} candles of {symbol} {timeframe} Futures data...")
        # Notice the symbol format "BTC/USDT:USDT" - this is ccxt's standard for linear perpetuals
        bars = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def fetch_open_interest(self, symbol="BTC/USDT:USDT", timeframe="1h", limit=1000):
        """Fetches historical Open Interest to measure leverage in the system."""
        try:
            print(f"Fetching Open Interest for {symbol}...")
            # Note: Not all exchanges support historical OI via ccxt standard methods,
            # but Binance and Bybit generally support implicit futures calls.
            oi_data = self.exchange.fetch_open_interest_history(symbol, timeframe, limit=limit)
            
            # Extract just the timestamp and the OI value
            clean_oi = []
            for entry in oi_data:
                clean_oi.append({
                    'timestamp': pd.to_datetime(entry['timestamp'], unit='ms'),
                    'open_interest': entry['openInterestValue'] # The USD value of open contracts
                })
            
            return pd.DataFrame(clean_oi)
        
        except Exception as e:
            print(f"⚠️ Exchange {EXCHANGE_ID} might not support historical OI directly via ccxt: {e}")
            # Fallback to an empty dataframe so the bot doesn't crash
            return pd.DataFrame(columns=['timestamp', 'open_interest'])

    def get_full_market_data(self, symbol="BTC/USDT:USDT", timeframe="1h", limit=1000):
        """Merges Price Data with Open Interest into one master dataframe."""
        df_price = self.fetch_ohlcv(symbol, timeframe, limit)
        df_oi = self.fetch_open_interest(symbol, timeframe, limit)
        
        if not df_oi.empty:
            # Merge the two datasets matching their timestamps exactly
            df_merged = pd.merge(df_price, df_oi, on='timestamp', how='left')
            # Fill any missing OI gaps with the previous value
            df_merged['open_interest'] = df_merged['open_interest'].ffill().fillna(0)
            return df_merged
        else:
            # If OI fails, just return price with a zeroed OI column
            df_price['open_interest'] = 0.0
            return df_price