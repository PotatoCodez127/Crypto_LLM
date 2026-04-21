"""
Feature extraction module tailored for Machine Learning pipelines.
Focuses on stationarity, microstructure, and normalized features.
"""

import pandas as pd
import numpy as np

class FeatureExtractor:
    """Extracts ML-ready, stationary features from OHLCV + Derivatives data."""

    def __init__(self):
        pass

    def calculate_log_returns(self, prices):
        """Calculate stationary log returns (crucial for ML)."""
        return np.log(prices / prices.shift(1)).fillna(0)

    def calculate_cvd_approximation(self, df):
        """
        Approximates Cumulative Volume Delta (CVD) using bar price action.
        (True CVD requires tick data, this is the standard bar-level approximation).
        """
        # Calculate the proportion of the candle that was bullish vs bearish
        # (close - open) / (high - low) gives a ratio from -1 (full sell) to 1 (full buy)
        range_hl = df['high'] - df['low']
        # Prevent division by zero
        range_hl = range_hl.replace(0, np.nan) 
        
        delta_ratio = (df['close'] - df['open']) / range_hl
        bar_delta = df['volume'] * delta_ratio.fillna(0)
        
        return bar_delta.cumsum()

    def calculate_atr(self, df, window=14):
        """Calculate Average True Range (Volatility representation)."""
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift())
        low_close = abs(df["low"] - df["close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean().fillna(0)

    def z_score_normalize(self, series, window=100):
        """
        Rolling Z-Score to normalize features for ML ingestion.
        Ensures the model sees "relative extremes" rather than absolute numbers.
        """
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        
        # Prevent division by zero
        rolling_std = rolling_std.replace(0, 1)
        return ((series - rolling_mean) / rolling_std).fillna(0)

    def extract_features(self, df):
        """
        Extract all Phase 1 features suitable for XGBoost/LightGBM.
        """
        feature_df = df.copy()

        # 1. Price Stationarity
        feature_df["log_return"] = self.calculate_log_returns(feature_df["close"])
        
        # 2. Volatility Metrics
        feature_df["atr"] = self.calculate_atr(feature_df)
        feature_df["atr_normalized"] = self.z_score_normalize(feature_df["atr"], window=100)
        
        # 3. Microstructure / Derivatives Edge
        feature_df["cvd"] = self.calculate_cvd_approximation(feature_df)
        feature_df["cvd_trend"] = feature_df["cvd"].diff(3) # 3-period change in aggressive flow
        
        # Ensure OI and Funding exist (from handler), then calculate derivatives momentum
        if 'open_interest' in feature_df.columns:
            # Prevent ZeroDivisionError by temporarily converting 0s to NaNs before pct_change
            feature_df['oi_change_pct'] = feature_df['open_interest'].replace(0, np.nan).pct_change().fillna(0)
            feature_df['oi_zscore'] = self.z_score_normalize(feature_df['open_interest'], window=50)
            
        if 'funding_rate' in feature_df.columns:
            feature_df['funding_zscore'] = self.z_score_normalize(feature_df['funding_rate'], window=50)

        # 4. Standard technicals (Z-scored for ML)
        feature_df["volume_zscore"] = self.z_score_normalize(feature_df["volume"], window=50)
        
        # Drop raw prices (optional, but raw prices cause overfitting in decision trees)
        # feature_df = feature_df.drop(columns=['open', 'high', 'low', 'close'])

        return feature_df.fillna(0)

if __name__ == "__main__":
    extractor = FeatureExtractor()
    print("ML Feature Extractor Initialized.")