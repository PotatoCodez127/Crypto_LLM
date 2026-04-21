import sys
import os
import pandas as pd
import numpy as np

# Ensure Python can find the offline_training folder
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(project_root, 'offline_training'))

from strategy import get_signals

def test_feature_engineering_columns():
    """Tests if the V2 strategy correctly calculates institutional features."""
    
    # 1. Create a dummy DataFrame with 100 candles of fake data
    np.random.seed(42)
    df = pd.DataFrame({
        'open': np.random.uniform(50000, 51000, 100),
        'high': np.random.uniform(51000, 52000, 100),
        'low': np.random.uniform(49000, 50000, 100),
        'close': np.random.uniform(50000, 51000, 100),
        'volume': np.random.uniform(10, 100, 100)
    })
    
    # 2. Run the strategy
    processed_df = get_signals(df)
    
    # 3. Assertions (The Test)
    # Check if our new columns were successfully created
    expected_columns = ['log_ret', 'volatility_20', 'cvd_20', 'z_score_50', 'raw_signal', 'signal', 'atr']
    for col in expected_columns:
        assert col in processed_df.columns, f"Missing crucial column: {col}"
        
    # Check that it didn't drop any rows (should still be 100)
    assert len(processed_df) == 100, "Strategy function dropped rows!"
    
    # Check that signals are strictly 1, 0, or -1
    valid_signals = {1, 0, -1}
    unique_signals = set(processed_df['signal'].unique())
    assert unique_signals.issubset(valid_signals), f"Invalid signal generated: {unique_signals}"