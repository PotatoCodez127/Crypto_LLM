import sys
import os
import pandas as pd
from unittest.mock import patch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.data_feed.handler import DataHandler

@patch('src.data_feed.handler.DataHandler.fetch_ohlcv')
@patch('src.data_feed.handler.DataHandler.fetch_open_interest')
def test_get_full_market_data(mock_fetch_oi, mock_fetch_ohlcv):
    """Tests if the handler successfully merges price data with Open Interest."""
    
    # 1. Create fake Price Data (returned by the mock)
    mock_fetch_ohlcv.return_value = pd.DataFrame({
        'timestamp': pd.to_datetime(['2026-01-01 00:00:00', '2026-01-01 01:00:00']),
        'close': [50000, 51000]
    })
    
    # 2. Create fake Open Interest Data (returned by the mock)
    mock_fetch_oi.return_value = pd.DataFrame({
        'timestamp': pd.to_datetime(['2026-01-01 00:00:00', '2026-01-01 01:00:00']),
        'open_interest': [1000000, 1500000]
    })
    
    handler = DataHandler()
    
    # 3. Call the master function (it will use our fake data instead of ccxt)
    merged_df = handler.get_full_market_data()
    
    # 4. Assertions
    assert 'close' in merged_df.columns
    assert 'open_interest' in merged_df.columns
    assert merged_df['open_interest'].iloc[1] == 1500000, "OI data did not merge correctly"