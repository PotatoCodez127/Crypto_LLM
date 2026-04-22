"""
build_features.py
Transforms raw OHLCV into an Institutional V2 Feature Matrix.
"""

import pandas as pd
import numpy as np
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "data", "btc_1h_3y.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "btc_1h_3y_v2.csv")

def engineer_features():
    print("🔬 INITIALIZING V2 DATA ENGINEERING PIPELINE...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Raw data not found at {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    
    # 1. Price Action & Returns
    print("   [+] Calculating Log Returns...")
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # 2. Cumulative Volume Delta (CVD) Proxy
    # We assign volume as positive (buying pressure) if candle is green, negative if red.
    print("   [+] Synthesizing Order Flow (CVD)...")
    df['candle_dir'] = np.where(df['close'] >= df['open'], 1, -1)
    df['volume_delta'] = df['volume'] * df['candle_dir']
    df['cvd'] = df['volume_delta'].cumsum()
    # Detrended CVD: Is buying pressure surging compared to the last 24 hours?
    df['cvd_trend'] = df['cvd'] - df['cvd'].rolling(window=24).mean() 

    # 3. Institutional Volatility (ATR Proxy)
    print("   [+] Calculating Volatility Normalizers (ATR)...")
    df['tr0'] = abs(df['high'] - df['low'])
    df['tr1'] = abs(df['high'] - df['close'].shift(1))
    df['tr2'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    df['atr_14'] = df['tr'].rolling(window=14).mean()
    df.drop(['tr0', 'tr1', 'tr2', 'tr'], axis=1, inplace=True) # Cleanup

    # 4. Stationarity (Z-Scores)
    # Machine Learning models (and logical rules) hate raw prices. They love standard deviations.
    print("   [+] Normalizing Time-Series (Z-Scores)...")
    df['close_zscore_50'] = (df['close'] - df['close'].rolling(50).mean()) / df['close'].rolling(50).std()
    df['volume_zscore_24'] = (df['volume'] - df['volume'].rolling(24).mean()) / df['volume'].rolling(24).std()

    # 5. Clean and Save
    df.dropna(inplace=True) # Drop the initial rows that have NaNs from rolling windows
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✅ SUCCESS! V2 Dataset saved to: {OUTPUT_FILE}")
    print(f"📊 New Features Available: cvd_trend, atr_14, close_zscore_50, volume_zscore_24")

if __name__ == "__main__":
    engineer_features()