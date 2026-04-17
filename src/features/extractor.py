"""
Feature extraction module for transforming market data into feature vectors.
"""

import pandas as pd
import numpy as np


class FeatureExtractor:
    """Extracts technical indicators and features from OHLCV data."""

    def __init__(self):
        """Initialize feature extractor with default parameters."""
        pass

    def calculate_rsi(self, prices, window=14):
        """
        Calculate Relative Strength Index.

        Args:
            prices (pd.Series): Price series
            window (int): Lookback window

        Returns:
            pd.Series: RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices, fast_period=12, slow_period=26, signal_period=9):
        """
        Calculate MACD indicator.

        Args:
            prices (pd.Series): Price series
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            signal_period (int): Signal line period

        Returns:
            tuple: (MACD line, Signal line, Histogram)
        """
        exp1 = prices.ewm(span=fast_period).mean()
        exp2 = prices.ewm(span=slow_period).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period).mean()
        histogram = macd - signal
        return macd, signal, histogram

    def calculate_atr(self, df, window=14):
        """
        Calculate Average True Range.

        Args:
            df (pd.DataFrame): DataFrame with 'high', 'low', 'close' columns
            window (int): Lookback window

        Returns:
            pd.Series: ATR values
        """
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift())
        low_close = abs(df["low"] - df["close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=window).mean()
        return atr

    def calculate_volume_change(self, volume, window=1):
        """
        Calculate volume change percentage.

        Args:
            volume (pd.Series): Volume series
            window (int): Period for change calculation

        Returns:
            pd.Series: Volume change percentage
        """
        vol_change = volume.pct_change(window).fillna(0)
        return vol_change

    def calculate_trend_slope(self, prices, window=20):
        """
        Calculate trend slope using linear regression.

        Args:
            prices (pd.Series): Price series
            window (int): Window for slope calculation

        Returns:
            pd.Series: Trend slopes
        """
        slopes = pd.Series(index=prices.index, dtype=float)
        for i in range(window, len(prices)):
            y = prices.iloc[i - window : i].values
            x = np.arange(len(y))
            if len(x) > 1:
                slope = np.polyfit(x, y, 1)[0]
                slopes.iloc[i] = slope
        return slopes.fillna(0)

    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """
        Calculate Bollinger Bands and bandwidth.

        Args:
            prices (pd.Series): Price series
            window (int): Lookback window
            num_std (int): Number of standard deviations

        Returns:
            tuple: (Upper band, Lower band, Bandwidth)
        """
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        bandwidth = (upper_band - lower_band) / rolling_mean
        return upper_band, lower_band, bandwidth

    def extract_features(self, df):
        """
        Extract all features from OHLCV data.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data

        Returns:
            pd.DataFrame: DataFrame with original data plus feature columns
        """
        # Make a copy to avoid modifying original data
        feature_df = df.copy()

        # Calculate RSI
        feature_df["rsi"] = self.calculate_rsi(feature_df["close"])

        # Calculate MACD
        macd, signal, hist = self.calculate_macd(feature_df["close"])
        feature_df["macd"] = macd
        feature_df["macd_signal"] = signal
        feature_df["macd_histogram"] = hist

        # Calculate ATR
        feature_df["atr"] = self.calculate_atr(feature_df)

        # Calculate volume change
        feature_df["volume_change"] = self.calculate_volume_change(feature_df["volume"])

        # Calculate trend slope
        feature_df["trend_slope"] = self.calculate_trend_slope(feature_df["close"])

        # Calculate Bollinger Band width
        _, _, bb_width = self.calculate_bollinger_bands(feature_df["close"])
        feature_df["bb_width"] = bb_width

        # Fill NaN values
        feature_df = feature_df.fillna(0)

        return feature_df

    def get_feature_vector(self, df, index=-1):
        """
        Get feature vector for a specific row.

        Args:
            df (pd.DataFrame): DataFrame with features
            index (int): Row index (default: -1 for latest)

        Returns:
            dict: Feature vector as dictionary
        """
        feature_df = self.extract_features(df)
        if len(feature_df) == 0:
            return {}

        row = feature_df.iloc[index]
        return {
            "rsi": row["rsi"],
            "macd": row["macd"],
            "atr": row["atr"],
            "volume_change": row["volume_change"],
            "trend_slope": row["trend_slope"],
            "bb_width": row["bb_width"],
        }


if __name__ == "__main__":
    # Example usage
    extractor = FeatureExtractor()
    # This would typically be used with actual OHLCV data
    print("Feature extractor initialized")
