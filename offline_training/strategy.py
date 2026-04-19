import pandas as pd
import numpy as np

def get_signals(df):
    """
    Simple Trend-Following Strategy:
    - Long (1) when 20 EMA > 50 EMA and RSI > 50
    - Short (-1) when 20 EMA < 50 EMA and RSI < 50
    """
    # 1. Calculate Indicators
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # Simple RSI calculation
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # 2. Generate Signals
    df['signal'] = 0  # Default to Neutral
    
    # Long Condition
    df.loc[(df['ema_20'] > df['ema_50']) & (df['rsi'] > 55), 'signal'] = 1
    
    # Short Condition
    df.loc[(df['ema_20'] < df['ema_50']) & (df['rsi'] < 45), 'signal'] = -1
    
    return df