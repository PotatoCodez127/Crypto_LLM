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
        return np.log(prices / prices.shift(1)).fillna(0)

    def calculate_cvd_approximation(self, df):
        range_hl = df['high'] - df['low']
        range_hl = range_hl.replace(0, np.nan) 
        delta_ratio = (df['close'] - df['open']) / range_hl
        bar_delta = df['volume'] * delta_ratio.fillna(0)
        return bar_delta.cumsum()

    def calculate_atr(self, df, window=14):
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift())
        low_close = abs(df["low"] - df["close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean().fillna(0)

    def z_score_normalize(self, series, window=100):
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        rolling_std = rolling_std.replace(0, 1)
        return ((series - rolling_mean) / rolling_std).fillna(0)

    def calculate_rsi(self, series, window=14):
        """Calculates Relative Strength Index for AI Swarm requests."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss.replace(0, 1) # Prevent div zero
        return 100 - (100 / (1 + rs))

    def calculate_bollinger_bands(self, series, window=20, num_std=2):
        """Calculates Bollinger Bands (Upper and Lower)."""
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band

    def extract_features(self, df):
        """Extracts Phase 1 features, preserving OHLC for State Machine."""
        feature_df = df.copy()

        # 1. Price Stationarity
        feature_df["log_return"] = self.calculate_log_returns(feature_df["close"])
        feature_df["log_return_lag_1"] = feature_df["log_return"].shift(1)
        feature_df["log_return_lag_2"] = feature_df["log_return"].shift(2)
        feature_df["log_return_lag_3"] = feature_df["log_return"].shift(3)
        
        # 2. Volatility Metrics
        feature_df["atr_14"] = self.calculate_atr(feature_df, window=14)
        feature_df["atr_normalized"] = self.z_score_normalize(feature_df["atr_14"], window=100)
        
        # 3. Microstructure Edge
        feature_df["cvd"] = self.calculate_cvd_approximation(feature_df)
        feature_df["cvd_trend"] = feature_df["cvd"].diff(3) 
        
        # 4. Standard Technicals (Fulfills Swarm Agent Requests)
        feature_df['rsi_14'] = self.calculate_rsi(feature_df['close']).fillna(50)
        ema_12 = feature_df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = feature_df['close'].ewm(span=26, adjust=False).mean()
        feature_df['macd_line'] = (ema_12 - ema_26).fillna(0)
        
        # 🔥 FIX: Add Bollinger Bands for the AI
        bb_upper, bb_lower = self.calculate_bollinger_bands(feature_df['close'])
        feature_df['bb_upper'] = bb_upper.fillna(0)
        feature_df['bb_lower'] = bb_lower.fillna(0)
        
        feature_df["volume_zscore_24"] = self.z_score_normalize(feature_df["volume"], window=24)
        feature_df["close_zscore_50"] = self.z_score_normalize(feature_df["close"], window=50)
        
        # 5. CRYPTO-NATIVE MULTI-TIMEFRAME ANALYSIS (MTFA)
        feature_df['macro_return_24h'] = feature_df['close'].pct_change(24).fillna(0)
        feature_df['macro_return_96h'] = feature_df['close'].pct_change(96).fillna(0)
        feature_df['macro_atr_24h'] = feature_df['atr_14'].rolling(24).mean().fillna(0)
        feature_df['macro_trend_accel'] = feature_df['macro_return_24h'].diff(3).fillna(0)
        feature_df['macro_trend_zscore'] = self.z_score_normalize(feature_df['macro_return_24h'], window=100)

        return feature_df.fillna(0)