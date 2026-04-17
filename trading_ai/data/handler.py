"""
Data handler module for fetching and storing cryptocurrency market data.
"""

import ccxt
import pandas as pd
from datetime import datetime
import os
from config.settings import EXCHANGE_NAME, TIMEFRAME, SYMBOL, DATA_STORAGE_PATH


class DataHandler:
    """Handles data fetching and storage for the trading system."""

    def __init__(self):
        """Initialize the data handler with exchange connection."""
        self.exchange = getattr(ccxt, EXCHANGE_NAME)()
        self.data_file = DATA_STORAGE_PATH

        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)

    def fetch_ohlcv(self, symbol=None, timeframe=None, since=None, limit=1000):
        """
        Fetch OHLCV data from the exchange.

        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe for data
            since (int): Timestamp to start fetching from
            limit (int): Number of candles to fetch

        Returns:
            list: OHLCV data
        """
        symbol = symbol or SYMBOL
        timeframe = timeframe or TIMEFRAME

        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            return ohlcv
        except Exception as e:
            print(f"Error fetching data: {e}")
            return []

    def ohlcv_to_dataframe(self, ohlcv_data):
        """
        Convert OHLCV data to pandas DataFrame.

        Args:
            ohlcv_data (list): Raw OHLCV data from exchange

        Returns:
            pd.DataFrame: Formatted DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        df = pd.DataFrame(
            ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def save_data(self, df):
        """
        Save DataFrame to CSV file.

        Args:
            df (pd.DataFrame): DataFrame to save
        """
        try:
            if os.path.exists(self.data_file):
                # Append new data to existing file
                existing_df = pd.read_csv(self.data_file)
                combined_df = pd.concat([existing_df, df]).drop_duplicates(
                    subset=["timestamp"]
                )
                combined_df.to_csv(self.data_file, index=False)
            else:
                # Create new file
                df.to_csv(self.data_file, index=False)

            print(f"Data saved to {self.data_file}")
        except Exception as e:
            print(f"Error saving data: {e}")

    def load_data(self):
        """
        Load data from CSV file.

        Returns:
            pd.DataFrame: Loaded data or empty DataFrame if file doesn't exist
        """
        try:
            if os.path.exists(self.data_file):
                df = pd.read_csv(self.data_file)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                return df
            else:
                print(f"Data file {self.data_file} not found.")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()

    def update_historical_data(self, days_back=1095):
        """
        Update historical data with specified number of days back.

        Args:
            days_back (int): Number of days to fetch historical data for
        """
        # Calculate milliseconds for the period
        since = int((datetime.now().timestamp() - days_back * 24 * 60 * 60) * 1000)

        # Fetch data in chunks to avoid limit restrictions
        all_data = []
        chunk_limit = 1000
        current_since = since

        while True:
            chunk = self.fetch_ohlcv(since=current_since, limit=chunk_limit)
            if not chunk or len(chunk) == 0:
                break

            all_data.extend(chunk)

            # Move to next chunk
            current_since = chunk[-1][0] + 1  # Next timestamp

            # If we got less than the limit, we've reached the end
            if len(chunk) < chunk_limit:
                break

        if all_data:
            df = self.ohlcv_to_dataframe(all_data)
            self.save_data(df)
            print(f"Fetched and saved {len(df)} records")
        else:
            print("No data fetched")

    def get_latest_candle(self):
        """
        Get the latest candle data.

        Returns:
            pd.DataFrame: Latest candle data
        """
        ohlcv = self.fetch_ohlcv(limit=1)
        if ohlcv:
            return self.ohlcv_to_dataframe(ohlcv)
        return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    handler = DataHandler()
    handler.update_historical_data(30)  # Update with last 30 days of data
