"""
train.py - The Strategy Sandbox
The AI agent edits this file to maximize FINAL_RESULT.
"""
from prepare import evaluate_strategy
import pandas as pd
import numpy as np

def get_signals(df):
    """
    Enhanced strategy using MACD crossover, RSI, and ATR volatility filter.
    """
    # Calculate EMAs for trend
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['signal_line']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR for volatility filter
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    df['atr_pct'] = df['atr'] / df['close']
    
    # Generate signals
    df['signal'] = 0
    
    # Long conditions: MACD crosses above signal line, RSI > 50, volatility not too high
    long_condition = (
        (df['macd_diff'] > 0) & 
        (df['macd_diff'].shift(1) <= 0) &  # crossover up
        (df['rsi'] > 50) &
        (df['atr_pct'] < 0.05)  # ATR less than 5% of price
    )
    
    # Short conditions: MACD crosses below signal line, RSI < 50, volatility filter
    short_condition = (
        (df['macd_diff'] < 0) & 
        (df['macd_diff'].shift(1) >= 0) &  # crossover down
        (df['rsi'] < 50) &
        (df['atr_pct'] < 0.05)
    )
    
    # Exit conditions: opposite crossover or RSI extreme
    exit_long = (df['macd_diff'] < 0) & (df['macd_diff'].shift(1) >= 0)
    exit_short = (df['macd_diff'] > 0) & (df['macd_diff'].shift(1) <= 0)
    
    # Initialize position
    position = 0
    signals = []
    for i in range(len(df)):
        if long_condition.iloc[i] and position != 1:
            position = 1
        elif short_condition.iloc[i] and position != -1:
            position = -1
        elif exit_long.iloc[i] and position == 1:
            position = 0
        elif exit_short.iloc[i] and position == -1:
            position = 0
        signals.append(position)
    
    df['signal'] = signals
    return df

if __name__ == "__main__":
    # Monkey-patch strategy module
    import strategy
    strategy.get_signals = get_signals
    
    # Evaluate strategy for 1-year period (matches prepare.py's default)
    final_score = evaluate_strategy('1y')
    
    # Output the exact string the AI is looking for
    print(f"FINAL_RESULT:{final_score}")
