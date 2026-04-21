import pandas as pd
import numpy as np

def get_signals(df):
    """
    MOMENTUM BREAKOUT STRATEGY:
    - Faster moving average crossover (fast=8, slow=24) for trend detection.
    - RSI (14) with dynamic thresholds based on recent volatility.
    - No ADX or low-volatility filters to increase signal frequency.
    - Fixed stop-loss at 3*ATR, take-profit at 5*ATR.
    - Cooldown period of 5 bars after exit to avoid whipsaw.
    - Position sizing: full size (1.0).
    """
    # 1. Trend: faster moving averages
    df['ma_fast'] = df['close'].rolling(window=8).mean()
    df['ma_slow'] = df['close'].rolling(window=24).mean()
    df['trend_up'] = df['ma_fast'] > df['ma_slow']
    df['trend_down'] = df['ma_fast'] < df['ma_slow']
    
    # 2. RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 3. Dynamic RSI thresholds based on recent volatility
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std()
    # Normalize volatility to adjust thresholds
    vol_mean = df['volatility'].rolling(window=100).mean()
    vol_adj = df['volatility'] / (vol_mean + 1e-8)
    # Lower threshold becomes more aggressive (higher) when volatility is high
    rsi_lower = 30 + 10 * np.tanh(vol_adj - 1.0)  # ranges ~20-40
    rsi_upper = 70 - 10 * np.tanh(vol_adj - 1.0)  # ranges ~60-80
    df['rsi_lower'] = rsi_lower
    df['rsi_upper'] = rsi_upper
    
    # 4. ATR for stops
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean().bfill().fillna(0)
    
    # 5. Entry conditions (no extra filters)
    long_entry = (df['trend_up']) & (df['rsi'] < df['rsi_lower'])
    short_entry = (df['trend_down']) & (df['rsi'] > df['rsi_upper'])
    
    # 6. Raw signals (no cooldown yet)
    df['raw_signal'] = 0
    df.loc[long_entry, 'raw_signal'] = 1
    df.loc[short_entry, 'raw_signal'] = -1
    
    # 7. Position management with fixed stop/target and cooldown
    df['signal'] = 0
    position = 0
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    cooldown = 0
    
    for i in range(len(df)):
        raw = df['raw_signal'].iloc[i]
        close = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        
        if cooldown > 0:
            cooldown -= 1
            df.iloc[i, df.columns.get_loc('signal')] = 0
            continue
        
        if position == 0:
            if raw == 1:
                position = 1
                entry_price = close
                stop_price = close - 3.0 * atr
                target_price = close + 5.0 * atr
                df.iloc[i, df.columns.get_loc('signal')] = 1.0
            elif raw == -1:
                position = -1
                entry_price = close
                stop_price = close + 3.0 * atr
                target_price = close - 5.0 * atr
                df.iloc[i, df.columns.get_loc('signal')] = -1.0
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 0
        elif position == 1:
            # Check exit conditions
            if close <= stop_price or close >= target_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
                cooldown = 5  # bars of no new position
                # Do not re-enter immediately
            else:
                df.iloc[i, df.columns.get_loc('signal')] = 1.0
        elif position == -1:
            if close >= stop_price or close <= target_price:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = 0
                cooldown = 5
            else:
                df.iloc[i, df.columns.get_loc('signal')] = -1.0
    
    return df
