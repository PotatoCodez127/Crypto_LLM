import pandas as pd
import numpy as np

def get_signals(df):
    """
    Enhanced Trend-Following Strategy with ATR trailing stop-loss.
    - Long (1) when 20 EMA > 50 EMA and RSI > 60
    - Short (-1) when 20 EMA < 50 EMA and RSI < 40
    - Apply trailing stop-loss based on 2*ATR(14)
    """
    # 1. Calculate Indicators
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # RSI calculation
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR calculation
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    df['atr'] = df['atr'].fillna(method='bfill').fillna(0)

    # 2. Generate raw signals with stricter thresholds
    df['raw_signal'] = 0
    df.loc[(df['ema_20'] > df['ema_50']) & (df['rsi'] > 60), 'raw_signal'] = 1
    df.loc[(df['ema_20'] < df['ema_50']) & (df['rsi'] < 40), 'raw_signal'] = -1

    # 3. Apply trailing stop-loss
    df['signal'] = 0
    position = 0  # 0: flat, 1: long, -1: short
    stop_price = 0.0
    atr_multiplier = 2.0

    for i in range(len(df)):
        # Current values
        raw = df['raw_signal'].iloc[i]
        close = df['close'].iloc[i]
        atr = df['atr'].iloc[i]

        # Entry logic
        if position == 0:
            if raw == 1:
                position = 1
                stop_price = close - atr_multiplier * atr
                df.iloc[i, df.columns.get_loc('signal')] = 1
            elif raw == -1:
                position = -1
                stop_price = close + atr_multiplier * atr
                df.iloc[i, df.columns.get_loc('signal')] = -1
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 0
        # Manage existing position
        elif position == 1:
            # Update trailing stop (only move up)
            new_stop = close - atr_multiplier * atr
            if new_stop > stop_price:
                stop_price = new_stop
            # Check stop loss
            if close <= stop_price:
                # Exit long
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 1
        elif position == -1:
            new_stop = close + atr_multiplier * atr
            if new_stop < stop_price:
                stop_price = new_stop
            if close >= stop_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = -1

    return df
