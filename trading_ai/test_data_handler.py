"""
Test script for the data handler module.
"""

from data.handler import DataHandler
import pandas as pd


def test_data_handler():
    """Test the data handler functionality."""
    print("Testing Data Handler...")

    # Initialize data handler
    handler = DataHandler()
    print("Data handler initialized")

    # Test fetching latest candle
    print("\nFetching latest candle...")
    latest_candle = handler.get_latest_candle()
    if not latest_candle.empty:
        print("Latest candle data:")
        print(latest_candle)
    else:
        print("Failed to fetch latest candle")

    # Test feature extraction
    if not latest_candle.empty:
        print("\nTesting data loading...")
        loaded_data = handler.load_data()
        if not loaded_data.empty:
            print(f"Loaded {len(loaded_data)} records from storage")
        else:
            print("No data loaded from storage")

    print("\nData handler test complete!")


if __name__ == "__main__":
    test_data_handler()
