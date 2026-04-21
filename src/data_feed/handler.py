import ccxt
import pandas as pd
from src.config.settings import EXCHANGE_ID

class DataHandler:
    def __init__(self):
        # Connect to the Exchange ANONYMOUSLY
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
                # 🔥 FIX: Fallback safely between Quote Value and Base Amount
                oi_val = entry.get('openInterestValue') or entry.get('openInterestAmount') or 0.0
                clean_oi.append({
                    'timestamp': pd.to_datetime(entry['timestamp'], unit='ms'),
                    'open_interest': float(oi_val)
                })
            return pd.DataFrame(clean_oi)
        except Exception as e:
            print(f"⚠️ Exchange {EXCHANGE_ID} OI fetch failed: {e}")
            return pd.DataFrame(columns=['timestamp', 'open_interest'])

    def fetch_funding_rates(self, symbol="BTC/USDT:USDT", limit=1000):
        try:
            print(f"Fetching Funding Rates for {symbol}...")
            # Note: Funding rates are usually 8h intervals, we will forward-fill them in the merge
            funding_data = self.exchange.fetch_funding_rate_history(symbol, limit=limit)
            
            clean_funding = []
            for entry in funding_data:
                clean_funding.append({
                    'timestamp': pd.to_datetime(entry['timestamp'], unit='ms'),
                    'funding_rate': entry['fundingRate']
                })
            return pd.DataFrame(clean_funding)
        except Exception as e:
            print(f"⚠️ Exchange {EXCHANGE_ID} Funding Rate fetch failed: {e}")
            return pd.DataFrame(columns=['timestamp', 'funding_rate'])

    def get_full_market_data(self, symbol="BTC/USDT:USDT", timeframe="1h", limit=1000):
        """Merges Price Data, Open Interest, and Funding Rates into one master dataframe."""
        df_price = self.fetch_ohlcv(symbol, timeframe, limit)
        df_oi = self.fetch_open_interest(symbol, timeframe, limit)
        df_funding = self.fetch_funding_rates(symbol, limit)
        
        # Merge Open Interest
        if not df_oi.empty:
            # Group by timestamp just in case CCXT returns duplicates
            df_oi = df_oi.groupby('timestamp').last().reset_index()
            df_merged = pd.merge(df_price, df_oi, on='timestamp', how='left')
            df_merged['open_interest'] = df_merged['open_interest'].ffill().fillna(0)
        else:
            df_merged = df_price
            df_merged['open_interest'] = 0.0

        # Merge Funding Rates (Merge on closest exact timestamp, then forward fill for the 1H bars in between 8H intervals)
        if not df_funding.empty:
            df_funding = df_funding.groupby('timestamp').last().reset_index()
            df_merged = pd.merge(df_merged, df_funding, on='timestamp', how='left')
            df_merged['funding_rate'] = df_merged['funding_rate'].ffill().fillna(0)
        else:
            df_merged['funding_rate'] = 0.0
            
        return df_merged