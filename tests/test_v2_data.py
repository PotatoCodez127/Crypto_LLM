import sys
import os
# Force Python to look in the parent directory (the root folder) for modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data_feed.handler import DataHandler
from src.features.extractor import FeatureExtractor

def run_pipeline_test():
    print("🚀 Initiating V2 Data Pipeline Test...\n")
    
    # 1. Initialize our new classes
    handler = DataHandler()
    extractor = FeatureExtractor()
    
    # 2. Fetch the merged OHLCV, Open Interest, and Funding Data
    # Using a smaller limit just to ensure the API endpoints connect and merge correctly
    print("📡 Requesting data from exchange...")
    raw_df = handler.get_full_market_data(symbol="BTC/USDT:USDT", timeframe="1h", limit=200)
    
    if raw_df.empty:
        print("❌ FAILED: DataHandler returned an empty DataFrame.")
        return
        
    print(f"✅ SUCCESS: Fetched {len(raw_df)} rows of raw market data.")
    print(f"Columns present: {list(raw_df.columns)}\n")
    
    # 3. Process through the ML Extractor
    print("🧮 Extracting ML-ready features (Stationarity, CVD, Z-Scores)...")
    ml_df = extractor.extract_features(raw_df)
    
    # 4. Verification output
    print("\n📊 --- FINAL ML DATAFRAME (Last 5 Rows) --- 📊")
    # Set pandas to show all columns so we can verify the new features exist
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    # Print the last 5 rows, dropping the raw OHLC to focus on the ML features
    columns_to_show = [col for col in ml_df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    print(ml_df[columns_to_show].tail())
    
    print("\n🔍 Checking for NaN values in ML features...")
    nan_counts = ml_df[columns_to_show].isna().sum()
    if nan_counts.sum() == 0:
        print("✅ SUCCESS: No missing values detected in feature set.")
    else:
        print("⚠️ WARNING: NaN values detected!")
        print(nan_counts[nan_counts > 0])

if __name__ == "__main__":
    run_pipeline_test()